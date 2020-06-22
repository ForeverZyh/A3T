# Robustness to Programmable String Transformations via Augmented Abstract Training

## Overview

This repository contains the code implementation of the paper *Robustness to Programmable String Transformations via Augmented Abstract Training*.

**A3T** is an adversarial training technique that combines the strengths of augmentation and abstraction techniques. The key idea underlying A3T is to decompose the perturbation space into two subsets, one that can be explored using augmentation and one that can be abstracted.

The structure of this repositoty:

- `DSL`: contains our domain specific lanauge for specifying the perturbation space. We also include a generalized version of HotFlip in this folder. 
- `dataset`: contains the dataset and data preprossing files. The csv files on AG dataset can be found at [this repo](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv). The training and validation set of SST2 dataset are downloaded automatically. The test set can be found at [this repo](https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv). The Glove word embedding can be found at [this website](http://nlp.stanford.edu/data/glove.6B.zip). The synonyms we used can be found at [this website](http://paraphrase.org/#/download), and we used English-Lexical-S Size.
- `diffai`: is a submodule containing the implementation of A3T built on top of diffai.
- `models`: contains the baselines of normal training, random augmentation, and HotFlip augmentation. 

We also uploaded our checkpoints [here](https://drive.google.com/file/d/1QCOAGNH7Fq3jWerTD5ArocOAILbG3OA3/view?usp=sharing).



## Environments 

We encourage users to use virtual environments such as virtualenv or conda.

```bash
cd diffai 
pip install -r requirements.txt 
cd ..
pip install tensorflow-gpu==1.13.1
pip install keras
pip install tensorflow-datasets==1.3.2
pip install nltk
```

Sometimes you may need to downgrade `numpy` to 1.16.1 or `Pillow` to 6.1.0 by

```bash
pip uninstall numpy && pip install numpy==1.16.1
pip uninstall Pillow && pip install Pillow==6.1.0
```



## Experiment Scripts

We provided the experiment setups of baselines in `exp_scripts_AG.py` and `exp_scripts_SST2.py`. And we show some examples of training and evluating the models.

- The following code segment does normal training on AG dataset and then evaluates on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{SwapPair}, 2), (T_{SubAdj}, 2)\}$.

```python
AG_train("char_AG") # train() is the normal training method
AG_test_model("char_AG", target_transformation="Composition(swap, swap, sub, sub)", truncate=35) # pass the target_transformation argument as the target perturbation space as well as the truncate length
AG_test_model("char_AG", func=partial(SwapSub(2,2)), truncate=35)  # using SwapSub to compute the exhaustive accuracy
```

- The following code segment does random augmentation training on AG dataset and then evaluates on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{Del}, 2), (T_{SubAdj}, 2)\}$.

```python
AG_adv_train("char_AG_aug", target_transformation="Composition(delete, delete, sub, sub)", adv_train_random=True, truncate=30) # adv_train() is the augmentation training method, adv_train_random=True means random augmentation, False means HotFlip augmentation
AG_test_model("char_AG_aug", target_transformation="Composition(delete, delete, sub, sub)", truncate=30)
AG_test_model("char_AG_aug", func=partial(DelDupSubChar(2,0,2)), truncate=30) # using DelDupSubChar to compute exhaustive accuracy containing Ins, Del, or both
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

- The following commands do A3T(HotFlip) training on AG dataset and then evaluate on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{InsAdj}, 2), (T_{SubAdj}, 2)\}$.

```bash
python ./diffai/. -d "Mix(a=Point(),b=Box(),aw=1,bw=0)" -t "Point()" -t "Box()" -n CharLevelAGSub -D AG --epochs 10 --batch-size 20 --test-first True --test-size=1000 --decay-fir=True --train-delta=2 --adv-train=2 --transform='Composition(ins, ins)' --train-ratio=0.5 --epoch_ratio=0.8 --truncate=30
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D AG --width 0 --test-size=7600 --adv-test=True --transform='Composition(ins, ins, sub, sub)' --truncate=30
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D AG --width 0 --test-size=7600 --test-func='DelDupSubChar(0,2,2,d,truncate=30)' --truncate=30
```

- The following code segment does A3T(search) training on SST2 dataset and then evaluates on HotFlip accuracy and exhaustive accuracy against perturbation $\{(T_{DelStop}, 2), (T_{SubSyn}, 2)\}$.

```bash
python ./diffai/. -d "Mix(a=Point(),b=Box(),aw=1,bw=0)" -t "Point()" -t "Box()" -n WordLevelSST2 -D SST2 --epochs 20 --batch-size 40 --test-first True --test-size=1821 --decay-fir=True --train-delta=2 --e-train=2 --test-func='DelDupSubWord(2,0,0,d)' -r 0.005
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D SST2 --width 0 --test-size=1821 --adv-test=True --transform='Composition(delete, delete, sub, sub)'
test-diffai -t Point --test TARGET_PYNET --test-batch-size 1 -D SST2 --width 0 --test-size=1821 --test-func='DelDupSubWord(2,0,2,d)
```

## Published Work

Yuhao Zhang, Aws Albarghouthi, Loris D’Antoni, Robustness to Programmable String Transformations via Augmented Abstract Training.

