# Robustness to Programmable String Transformations via Augmented Abstract Training

## Overview

This repository contains the code implementation of the paper *Robustness to Programmable String Transformations via Augmented Abstract Training*.

The structure of this repositoty:

- `DSL`: contains our domain specifc lanauge for specifying the perturbation space. We also include a generalized version of HotFlip in this folder. 

- `dataset`: contains the dataset and data preprossing files. The csv files on AG dataset can be found at [this repo](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv). The training and validation set of SST2 dataset are downloaded automatically. The test set can be found at [this repo](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv). The Glove word embedding can be found at [this website](http://nlp.stanford.edu/data/glove.6B.zip). The synonyms we used can be found at [this website](http://paraphrase.org/#/download), and we used English-Lexical-S Size.
- `diffai`: is a submodule containing the implementation of A3T built on top of diffai.
- `models`: contains the baselines of normal training, random augmentation, and HotFlip augmentation. 



## Environments 

We encourage users to use virtual environments such as virtualenv or conda.

```bash
cd diffai 
pip install -r requirements.txt 
cd ..
pip install tensorflow-gpu==1.13.1
pip install keras
pip install tensorflow-datasets
```

Sometimes you may need to downgrade `numpy` to 1.16.1 or `Pillow` to 6.1.0 by

```bash
pip uninstall numpy && pip install numpy==1.16.1
pip uninstall Pillow && pip uninstall Pillow==6.1.0
```



## Experiment Scripts

We provided the experiment setups of baselines in `exp_scripts_AG.py` and `exp_scripts_SST2.py`. And we show some examples of training and evluating the models.

- The following code segment does normal training on AG dataset and then evaluates on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{SwapPair}, 1), (T_{SubAdj}, 1)\}$.

```python
AG_train("char_AG") # train() is the normal training method
AG_test_model("char_AG", target_transformation="Composition(swap, sub)") # pass the target_transformation argument as the target perturbation space
AG_test_model("char_AG", func=partial(SwapSub(1,1)))  # using SwapSub to compute the exhaustive accuracy containing Swap
```

- The following code segment does random augmentation training on AG dataset and then evaluates on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{Del}, 1), (T_{SubAdj}, 1)\}$.

```python
AG_adv_train("char_AG_aug", target_transformation="Composition(delete, sub)", adv_train_random=True) # adv_train() is the augmentation training method, adv_train_random=True means random augmentation, False means HotFlip augmentation
AG_test_model("char_AG_aug", target_transformation="Composition(delete, sub)")
AG_test_model("char_AG_aug", func=partial(DelDupSubChar(1,0,1))) # using DelDupSubChar to compute exhaustive accuracy containing Ins, Del, or both
```

- The following code segment does HotFlip augmentation training on SST2 dataset and then evaluates on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{DelStop}, 2), (T_{Dup}, 2), (T_{SubSyn}, 2)\}$.

```python
SST2_adv_train("word_SST2_adv", target_transformation="Composition(delete, delete, ins, ins, sub, sub)") # adv_train() is the augmentation training method, adv_train_random=False means HotFlip augmentation
SST2_test_model("word_SST2_adv", target_transformation="Composition(delete, delete, ins, ins, sub, sub)")
SST2_test_model("word_SST2_adv", func=partial(DelDupSubWord(2,2,2))) # using DelDupSubWord for word-level
```

We provided the experiment setups of A3T(HotFlip) and A3T(search) in `exp_scripts.txt`. 

Notice in order to use command `test-diffai`, one has to first use the following command:

```bash
alias test-diffai="python ./diffai/. -d Point --epochs 1 --dont-write --test-freq 1"
```

And we show some examples of training and evluating the models.

- The following commands do A3T(HotFlip) training on AG dataset and then evaluate on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{InsAdj}, 1), (T_{SubAdj}, 1)\}$.

```bash
python ./diffai/. -d "Mix(a=Point(),b=Box(),aw=1,bw=0)" -t "Point()" -t "Box()" -n CharLevelAGSub -D AG --epochs 10 --batch-size 20 --test-first True --test-size=1000 --decay-fir=True --train-delta=1 --adv-train=2 --transform='ins' -r 0.005
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D AG --width 0 --test-size=7600 --adv-test=True --transform='Composition(ins, sub)'
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D AG --width 0 --test-size=7600 --text-func='DelDupSubChar(0,1,1,d)'
```

- The following code segment does A3T(search) training on SST2 dataset and then evaluates on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{DelStop}, 2), (T_{SubSyn}, 2)\}$.

```bash
python ./diffai/. -d "Mix(a=Point(),b=Box(),aw=1,bw=0)" -t "Point()" -t "Box()" -n WordLevelSST2 -D SST2 --epochs 20 --batch-size 40 --test-first True --test-size=1821 --decay-fir=True --train-delta=2 --e-train=2 --test-func='DelDupSubWord(2,0,0,d)' -r 0.005
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D SST2 --width 0 --test-size=1821 --adv-test=True --transform='Composition(delete, delete, sub, sub)'
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D SST2 --width 0 --test-size=1821 --text-func='DelDupSubWord(2,0,2,d)
```

