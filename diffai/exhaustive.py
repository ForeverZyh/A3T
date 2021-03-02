import numpy as np
import itertools
import torch

from utils import swap_pytorch
from dataset.dataset_loader import SSTWordLevel, Glove
from nltk import pos_tag
from DSL.Alphabet import Alphabet
import diffai.scheduling as S


def SwapSub(a, b, x, is_numpy=False, batch_size=64, truncate=None):
    adjacent_keys = S.Info.adjacent_keys
    if not is_numpy:
        x = x.cpu()
        X = []
    else:
        X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
        current_id = 0
    if truncate is None:
        truncated_len = len(x)
    else:
        truncated_len = truncate
    valid_swap_poss = [i for i in range(truncated_len - 1) if int(x[i]) != int(x[i + 1])]
    for swap in range(a, -1, -1):
        for swap_poss in itertools.combinations(tuple(valid_swap_poss), swap):
            # precheck whether overlape
            overlape = False
            for i in range(len(swap_poss) - 1):
                if swap_poss[i + 1] - swap_poss[i] == 1:
                    overlape = True
            if overlape:
                continue
            valid_sub_poss = [i for i in range(truncated_len) if (i not in swap_poss) and (i - 1 not in swap_poss) and len(adjacent_keys[int(x[i])]) > 0]
            for sub in range(b, -1, -1):
                for sub_poss in itertools.combinations(tuple(valid_sub_poss), sub):
                    if is_numpy:
                        x2 = X[current_id]
                        for swap_pos in swap_poss:
                            x2[swap_pos], x2[swap_pos + 1] = x2[swap_pos + 1], x2[swap_pos]
                    else:
                        x2 = x.clone()
                        for swap_pos in swap_poss:
                            swap_pytorch(x2, swap_pos, swap_pos + 1)
                    for sub_pos in sub_poss:
                        x2[sub_pos] = adjacent_keys[int(x[sub_pos])][0]
                    if is_numpy:
                        current_id += 1
                        if current_id >= batch_size:
                            yield X
                            X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
                            current_id = 0
                    else:
                        X.append(x2.unsqueeze(0))
                        if len(X) == batch_size:
                            yield torch.cat(X, 0).cuda()
                            X = []
    if len(X) > 0:
        if is_numpy:
            yield X
        else:
            yield torch.cat(X, 0).cuda()


def DelDupSubChar(a, b, c, x, is_numpy=False, batch_size=64, padding_id=0, truncate=None):
    adjacent_keys = S.Info.adjacent_keys
    if not is_numpy:
        x = x.cpu()
        X = []
    else:
        X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
        current_id = 0
    end_pos = len(x)
    while end_pos > 0 and int(x[end_pos - 1]) == padding_id:
        end_pos -= 1
    if truncate is None:
        truncated_len = end_pos
    else:
        truncated_len = min(end_pos, truncate)
    valid_sub_poss = [i for i in range(truncated_len) if len(adjacent_keys[int(x[i])]) > 0]
    for sub in range(c, -1, -1):
        for sub_poss in itertools.combinations(tuple(valid_sub_poss), sub):
            sub_pos_strs = []
            for sub_pos in sub_poss:
                sub_pos_strs.append(adjacent_keys[int(x[sub_pos])])
            for sub_pos_str in itertools.product(*sub_pos_strs):
                if is_numpy:
                    x3 = x.copy()
                else:
                    x3 = x.clone()
                for i, sub_pos in enumerate(sub_poss):
                    x3[sub_pos] = sub_pos_str[i]
                valid_dup_poss = [i for i in range(truncated_len) if i not in sub_poss and len(adjacent_keys[int(x[i])]) > 0]
                for dup in range(b, -1, -1):
                    for dup_poss in itertools.combinations(tuple(valid_dup_poss), dup):
                        valid_del_poss = [i for i in range(truncated_len) if (i not in dup_poss) and (i not in sub_poss)]
                        for delete in range(a, -1, -1):
                            for del_poss in itertools.combinations(tuple(valid_del_poss), delete):
                                if is_numpy:
                                    x2 = X[current_id]
                                else:
                                    x2 = x.clone()
                                copy_point = 0
                                paste_point = 0
                                while copy_point < end_pos and paste_point < end_pos:
                                    if copy_point in dup_poss:
                                        x2[paste_point] = x3[copy_point]
                                        paste_point += 1
                                        if paste_point < end_pos:
                                            x2[paste_point] = adjacent_keys[int(x3[copy_point])][0]
                                            paste_point += 1
                                            copy_point += 1
                                    elif copy_point in del_poss:
                                        copy_point += 1
                                    else:
                                        x2[paste_point] = x3[copy_point]
                                        paste_point += 1
                                        copy_point += 1

                                while paste_point < end_pos:
                                    x2[paste_point] = padding_id
                                    paste_point += 1

                                if is_numpy:
                                    current_id += 1
                                    if current_id >= batch_size:
                                        yield X
                                        X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
                                        current_id = 0
                                else:
                                    X.append(x2.unsqueeze(0))
                                    if len(X) == batch_size:
                                        yield torch.cat(X, 0).cuda()
                                        X = []

    if len(X) > 0:
        if is_numpy:
            yield X
        else:
            yield torch.cat(X, 0).cuda()
            

def DelDupSubWord(a, b, c, x, is_numpy=False, batch_size=64, del_set={"a", "and", "the", "of", "to"}, padding_id=0):
    SSTWordLevel.build()
    if not is_numpy:
        x = x.cpu()
        X = []
    else:
        X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
        current_id = 0
    end_pos = len(x)
    while end_pos > 0 and int(x[end_pos - 1]) == padding_id:
        end_pos -= 1
        
    valid_sub_poss = [i for i in range(end_pos) if int(x[i]) in SSTWordLevel.synonym_dict_id]
    input_pos_tag = pos_tag(Alphabet.to_string(x.long() if not is_numpy else x, True))
    for sub in range(c, -1, -1):
        for sub_poss in itertools.combinations(tuple(valid_sub_poss), sub):
            sub_pos_strs = []
            for sub_pos in sub_poss:
                sub_pos_strs.append([])
                for k in range(len(SSTWordLevel.synonym_dict_id[int(x[sub_pos])])):
                    if SSTWordLevel.synonym_dict_pos_tag[int(x[sub_pos])][k] == input_pos_tag[sub_pos][1]:
                        sub_pos_strs[-1].append(SSTWordLevel.synonym_dict_id[int(x[sub_pos])][k])
            for sub_pos_str in itertools.product(*sub_pos_strs):
                if is_numpy:
                    x3 = x.copy()
                else:
                    x3 = x.clone()
                for i, sub_pos in enumerate(sub_poss):
                    x3[sub_pos] = sub_pos_str[i]
                valid_dup_poss = [i for i in range(end_pos) if i not in sub_poss]
                for dup in range(b, -1, -1):
                    for dup_poss in itertools.combinations(tuple(valid_dup_poss), dup):
                        valid_del_poss = [i for i in range(end_pos) if (i not in dup_poss) and (i not in sub_poss) and Glove.id2str[int(x[i])] in del_set]
                        for delete in range(a, -1, -1):
                            for del_poss in itertools.combinations(tuple(valid_del_poss), delete):
                                if is_numpy:
                                    x2 = X[current_id]
                                else:
                                    x2 = x.clone()
                                copy_point = 0
                                paste_point = 0
                                while copy_point < end_pos and paste_point < end_pos:
                                    if copy_point in dup_poss:
                                        x2[paste_point] = x3[copy_point]
                                        paste_point += 1
                                        if paste_point < end_pos:
                                            x2[paste_point] = x3[copy_point]
                                            paste_point += 1
                                            copy_point += 1
                                    elif copy_point in del_poss:
                                        copy_point += 1
                                    else:
                                        x2[paste_point] = x3[copy_point]
                                        paste_point += 1
                                        copy_point += 1

                                while paste_point < end_pos:
                                    x2[paste_point] = padding_id
                                    paste_point += 1

                                if is_numpy:
                                    current_id += 1
                                    if current_id >= batch_size:
                                        yield X
                                        X = np.tile(np.expand_dims(x, 0), (batch_size, 1))
                                        current_id = 0
                                else:
                                    X.append(x2.unsqueeze(0))
                                    if len(X) == batch_size:
                                        yield torch.cat(X, 0).cuda()
                                        X = []

    if len(X) > 0:
        if is_numpy:
            yield X
        else:
            yield torch.cat(X, 0).cuda()
