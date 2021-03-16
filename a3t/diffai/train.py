import time as sys_time
from timeit import default_timer as timer
import torch.onnx
import torch.optim as optim
import os
import math
import warnings
from torch.serialization import SourceChangeWarning

from a3t.diffai.helpers import Timer
import a3t.diffai.helpers as h
import a3t.diffai.models as M
import a3t.diffai.goals as goals
import a3t.diffai.scheduling as S
from a3t.DSL.general_HotFlip import GeneralHotFlipAttack


# generate adv attack examples
def adv_batch(model, attack, batch_X, batch_Y, adv_num):
    adv_batch_X = []
    for x, y in zip(batch_X, batch_Y):
        ans = attack.gen_adv(model, model.to_strs(x), y.item(), adv_num, model.get_embed)
        for adv_x in ans:
            adv_batch_X.append(torch.LongTensor(model.to_ids(adv_x)).to(model.device).unsqueeze(0))
        for _ in range(1 + adv_num - len(ans)):
            adv_batch_X.append(x.unsqueeze(0))

    return torch.cat(adv_batch_X, 0), batch_Y.unsqueeze(-1).repeat((1, adv_num + 1)).view(-1)


def train(vocab, train_loader, val_loader, test_loader, adv_perturb, abs_perturb, args, fixed_len=None, num_classes=2
          , load_path=None, test=False):
    """
    training pipeline for A3T
    :param vocab: the vocabulary of the model, see dataset.dataset_loader.Vocab for details
    :param train_loader: the dataset loader for train set, obtained from a3t.diffai.helpers.loadDataset
    :param val_loader: the dataset loader for validation set
    :param test_loader: the dataset loader for test set
    :param adv_perturb: the perturbation space for HotFlip training
    :param abs_perturb: the perturbation space for abstract training
    :param args: the arguments for training
    :param fixed_len: CNN models need to pad the input to a certain length
    :param num_classes: the number of classification classes
    :param load_path: if specified, point to the file of loading net
    :param test: True if test, train otherwise
    """
    n = args.model_srt
    assert n in ["WordLevelSST2", "CharLevelSST2"]
    if test:
        assert load_path is not None
    m = getattr(M, n)
    args.log_interval = int(50000 / (args.batch_size * args.log_freq))
    domain = ["Mix(a=Point(),b=Box(),aw=1,bw=0)"]
    h.max_c_for_norm = args.max_norm

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed + 1)
    torch.cuda.manual_seed_all(args.seed + 2)

    input_dims = train_loader.dataset[0][0].size()

    print("input_dims: ", input_dims)
    print("Num classes: ", num_classes)
    vargs = vars(args)

    S.TrainInfo.total_batches_seen = 0
    decay_ratio_per_epoch = 1 / (args.epochs * args.epoch_perct_decay)

    ### Get model

    def buildNet(n):
        n = n(num_classes)
        n = n.infer(input_dims)
        if args.clip_norm:
            n.clip_norm()
        return n

    if test:
        def loadedNet():
            warnings.simplefilter("ignore", SourceChangeWarning)
            return torch.load(load_path)

        model = loadedNet().double() if h.dtype == torch.float64 else loadedNet().float()
    else:
        model = buildNet(m)
        model.__name__ = n

        print("Name: ", model.__name__)
        print("Number of Neurons (relus): ", model.neuronCount())
        print("Number of Parameters: ", sum([h.product(s.size()) for s in model.parameters() if s.requires_grad]))
        print("Depth (relu layers): ", model.depth())
        print()
        model.showNet()
        print()

    ### Get domain

    model = createModel(model, h.parseValues(domain, goals, S), h.catStrs(domain), args)
    for (a, b) in abs_perturb:
        assert a.length_preserving
    attack = GeneralHotFlipAttack(adv_perturb)
    victim_model = M.ModelWrapper(model, vocab, h.device, vargs, fixed_len)
    S.TrainInfo.abs_perturb = abs_perturb
    S.TrainInfo.victim_model = victim_model

    if not test:
        out_dir = os.path.join(args.out, n, h.file_timestamp())

        print("Saving to:", out_dir)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        print("Starting Training with:")
        with h.mopen(False, os.path.join(out_dir, "config.txt"), "w") as f:
            for k in sorted(vars(args)):
                h.printBoth("\t" + k + ": " + str(getattr(args, k)), f=f)
        print("")

        ### Prepare for training
        patience = args.early_stop_patience
        if patience > int(args.epochs * (1 - args.epoch_perct_decay)):
            warnings.warn("early stop patience is %d, but only %d epochs for full training" % (
                patience, int(args.epochs * (1 - args.epoch_perct_decay))), RuntimeWarning)
        last_best = -1
        best = 1e10
        S.TrainInfo.cur_ratio = 0

        with h.mopen(False, os.path.join(out_dir, "log.txt"), "w") as f:
            startTime = timer()
            for epoch in range(1, args.epochs + 1):
                if f is not None:
                    f.flush()
                h.printBoth("Elapsed-Time: {:.2f}s\n".format(timer() - startTime), f=f)
                is_best = False
                with Timer("train model in epoch", 1, f=f):
                    train_epoch(epoch, model, victim_model, attack, args, train_loader)
                    original_loss, robust_loss, pr_safe = test_epoch(model, victim_model, attack, args, val_loader,
                                                                     args.adv_train_num, f)
                    if S.TrainInfo.cur_ratio == 1:  # early stopping begins
                        if robust_loss < best:
                            best = robust_loss
                            last_best = epoch
                            is_best = True
                        elif epoch - last_best > patience:
                            h.printBoth("Early stopping at epoch %d\n" % epoch, f=f)
                            break
                    S.TrainInfo.cur_ratio = min(S.TrainInfo.cur_ratio + decay_ratio_per_epoch, 1)

                prepedname = model.ty.name.replace(" ", "_").replace(",", "").replace("(", "_").replace(")",
                                                                                                        "_").replace(
                    "=", "_")

                net_file = os.path.join(out_dir,
                                        model.name + "__" + prepedname + "_checkpoint_" + str(
                                            epoch) + "_with_{:1.3f}".format(
                                            pr_safe))

                h.printBoth("\tSaving netfile: {}\n".format(net_file + ".pynet"), f=f)

                if is_best or epoch % args.save_freq == 0:
                    print("Actually Saving")
                    torch.save(model.net, net_file + ".pynet")

            h.printBoth("Best at epoch %d\n" % last_best, f=f)
    else:
        ### Prepare for testing
        S.TrainInfo.cur_ratio = 1
        with Timer("test model", 1):
            test_epoch(model, victim_model, attack, args, test_loader, args.adv_test_num)


def train_epoch(epoch, model, victim_model, attack, args, train_loader):
    vargs = vars(args)
    model.train()

    print(("Cur ratio: {}").format(S.TrainInfo.cur_ratio))
    assert isinstance(model.ty, goals.DList) and len(model.ty.al) == 2
    for (i, a) in enumerate(model.ty.al):
        if not isinstance(a[0], goals.Point):
            model.ty.al[i] = (a[0], S.Const(args.train_lambda * S.TrainInfo.cur_ratio))
        else:
            model.ty.al[i] = (a[0], S.Const(1 - args.train_lambda * S.TrainInfo.cur_ratio))

    for batch_idx, (data, target) in enumerate(train_loader):
        S.TrainInfo.total_batches_seen += 1
        time = float(S.TrainInfo.total_batches_seen) / len(train_loader)
        data, target = data.to(h.device), target.to(h.device)

        model.global_num += data.size()[0]
        lossy = 0
        adv_time = sys_time.time()
        if args.adv_train_num > 0:
            data, target = adv_batch(victim_model, attack, data, target, args.adv_train_num)

        adv_time = sys_time.time() - adv_time

        timer = Timer("train a sample from " + model.name + " with " + model.ty.name, data.size()[0], False)
        with timer:
            for s in model.boxSpec(data.to_dtype(), target, time=time):
                model.optimizer.zero_grad()
                loss = model.aiLoss(*s, time=time, **vargs).mean(dim=0)
                lossy += loss.detach().item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    if p is not None and torch.isnan(p).any():
                        print("Such nan in vals")
                    if p is not None and p.grad is not None and torch.isnan(p.grad).any():
                        print("Such nan in postmagic")
                        stdv = 1 / math.sqrt(h.product(p.data.shape))
                        p.grad = torch.where(torch.isnan(p.grad),
                                             torch.normal(mean=h.zeros(p.grad.shape), std=stdv), p.grad)

                model.optimizer.step()

                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    if p is not None and torch.isnan(p).any():
                        print("Such nan in vals after grad")
                        stdv = 1 / math.sqrt(h.product(p.data.shape))
                        p.data = torch.where(torch.isnan(p.data),
                                             torch.normal(mean=h.zeros(p.data.shape), std=stdv), p.data)

                if args.clip_norm:
                    model.clip_norm()
                for p in model.parameters():
                    if not p.requires_grad:
                        continue
                    if p is not None and torch.isnan(p).any():
                        raise Exception("Such nan in vals after clip")

        model.addSpeed(timer.getUnitTime() + adv_time / len(data))

        if batch_idx % args.log_interval == 0:
            print((
                'Train Epoch {:12} Mix(a=Point(),b=Box(),aw=1,bw=0) {:3} [{:7}/{} ({:.0f}%)] \tAvg sec/ex {:1.8f}\tLoss: {:.6f}').format(
                model.name,
                epoch,
                batch_idx * len(data) // (args.adv_train_num + 1), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                model.speed,
                lossy))


def test_epoch(model, victim_model, attack, args, loader, adv_num, f=None):
    vargs = vars(args)

    robust_loss = 0
    original_loss = 0
    correct = 0
    width = 0
    safe = 0
    proved = 0
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(h.device), target.to(h.device)
        with torch.no_grad():
            loss = model.aiLoss(data, target, **vargs).sum(dim=0).item()
            pred = model(data).vanillaTensorPart().max(1, keepdim=True)[
                1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item() / len(loader.dataset)
            original_loss += loss / len(loader.dataset)

        if adv_num > 0:
            data, target = adv_batch(victim_model, attack, data, target, adv_num)

        with torch.no_grad():
            for s in model.boxSpec(data.to_dtype(), target):
                loss = model.aiLoss(*s, **vargs).sum(dim=0).item()
                robust_loss += loss / (adv_num + 1) / len(loader.dataset)  # sum up batch loss

            for s in model.boxSpec(data.to_dtype(), target):
                # ugly implementation, TODO: speed up test by inferring only once
                bs = model(s[0]).al[1].a
                org = model(data).vanillaTensorPart().max(1, keepdim=True)[1]
                width += bs.diameter().sum().item() / (adv_num + 1) / len(loader.dataset)
                proved += bs.isSafe(org).sum().item() / (adv_num + 1) / len(loader.dataset)
                safe += bs.isSafe(target).sum().item() / (adv_num + 1) / len(loader.dataset)

    h.printBoth((
        'Test: {:12} trained with Mix(a=Point(),b=Box(),aw=1,bw=0) - Avg sec/ex {:1.12f}, Accuracy: {}/{} ({:4.2f}%)').format(
        model.name,
        model.speed,
        round(correct * len(loader.dataset)), len(loader.dataset), correct * 100), f=f)

    pr_safe = safe
    pr_proved = proved
    pr_corr_given_proved = pr_safe / pr_proved if pr_proved > 0 else 0.0
    h.printBoth((
        "\tMix(a=Point(),b=Box(),aw=1,bw=0) - Width: {:<36.16f} Pr[Proved]={:<1.3f}  Pr[Corr and Proved]={:<1.3f}  Pr[Corr|Proved]={:<1.3f}  Robust Loss {:6f}  Original Loss {:6f}").format(
        width,
        pr_proved,
        pr_safe, pr_corr_given_proved,
        robust_loss, original_loss), f=f)

    return original_loss, robust_loss, pr_safe


def createModel(net, domain, domain_name, args):
    net_weights = net
    domain.name = domain_name

    m = {}
    for (k, v) in net_weights.state_dict().items():
        m[k] = v.to_dtype()
    net.load_state_dict(m)

    model = M.Top(args, net, domain)
    if args.clip_norm:
        model.clip_norm()
    if h.use_cuda:
        model.cuda()
    model.optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    model.lrschedule = optim.lr_scheduler.ReduceLROnPlateau(
        model.optimizer,
        'min',
        patience=args.lr_patience,
        threshold=args.threshold,
        min_lr=0.000001,
        factor=args.factor,
        verbose=True)

    net.name = net.__name__
    model.name = net.__name__

    return model
