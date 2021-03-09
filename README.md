# Robustness to Programmable String Transformations via Augmented Abstract Training

## Overview

This repository contains the code implementation of the paper *Robustness to Programmable String Transformations via Augmented Abstract Training*.

**A3T** is an adversarial training technique that combines the strengths of augmentation and abstraction techniques. The key idea underlying A3T is to decompose the perturbation space into two subsets, one that can be explored using augmentation and one that can be abstracted.

The structure of this repository:

- `DSL`: contains our domain specific language for specifying the perturbation space. We also include a generalized version of HotFlip in this folder. 
- `dataset`: prepares the SST2 dataset. 
The synonyms we used can be found at [this website](http://paraphrase.org/#/download), and we used English-Lexical-S Size.
The adjacent keyboard mapping can be found at [here](a3t/dataset/en.key).
- `diffai`: is a submodule containing the implementation of A3T built on top of diffai.

### Reproducing Results in the Paper

To reproduce results in the paper, please see the [artifacts tag](https://github.com/ForeverZyh/A3T/tags). 
The main branch of this repository is a library that supports general and easy usage of A3T with experiment scripts removed and code refactored. 
We also uploaded our checkpoints [here](https://drive.google.com/file/d/1QCOAGNH7Fq3jWerTD5ArocOAILbG3OA3/view?usp=sharing).


## Environments 
We encourage users to use virtual environments such as pyenv or conda.

### From `pip`

Get A3T using `pip`:

```bash
pip install a3t
```

### From source


```bash
git clone https://github.com/ForeverZyh/A3T.git
cd A3T
python setup.py install
```

### Troubleshooting
Sometimes you may need to downgrade `numpy` to 1.16.1 and/or `Pillow` to 6.1.0 by

```bash
pip uninstall numpy && pip install numpy==1.16.1
pip uninstall Pillow && pip install Pillow==6.1.0
```


## Get Started

We provide the training process of a word-level model and a char-level model on the SST2 dataset. 
Please see the `tests/test_run.py` for details.

### Prepare the Dataset and the Model

The default save directory is `/tmp/.A3T`, but one can also specify their own path (see `Glove.build()` in `a3t/dataset/dataset_loader.py`).

Use the following code to load the word-level model and sst2 word dataset:
```python
from a3t.dataset.dataset_loader import SST2WordLevel, Glove

# Load the Glove embedding 6B.50d
Glove.build(6, 50)
SST2WordLevel.build()
```

Use the following code the load the char-level model and sst2 char dataset:
```python
from a3t.dataset.dataset_loader import SST2CharLevel

SST2CharLevel.build()
```

The `loadDataset` in `a3t.diffai.helpers` can help to load the dataset. 
The method accepts four arguments (one optional)

```python
def loadDataset(dataset, batch_size, type, test_slice=None):
    """
    load the dataset
    :param dataset: the name of the dataset, currently support SST2CharLevel and SST2WordLevel
    :param batch_size: the batch size of the dataset loader
    :param type: "train", "val", "test"
    :param test_slice: select a slice of the data
    :return: a dataset loader
    """
```

### Customize the String Transformations

In general, A3T supports customized string transformations provided by the users.
The `DSL.transformation` contains several string transformations already defined and used in the experiments of the paper, namely,
`Sub`, `SubChar`, `Del`, `Ins`, `InsChar`, and `Swap`. Among those transformations, `Sub`, `SubChar`, and `Swap` are labeled as length-preserving transformations, which allows robust training.

One can define their own string transformations by implementing the abstract class `Transformation` and two functions `get_pos` and `transformer` as described in our paper. 
`get_pos` accepts a list of input tokens and returns a list of position pairs (start, end).
`transformer` accepts a list of input tokens and a start-end position pair and returns an iterator which enumerates the possible transformations at the start-end position.

#### Define a perturbation space

A perturbation space is in the form of `[(Trans_1, delta_1), ..., (Trans_n, delta_n)]`. Ideally, the perturbation is a set of the string transformations, but we use a list to store the perturbation space. 
In other words, we impose an order in the perturbation space, which will effect the HotFlip attack (see TODO in `GeneralHotFlipAttack.gen_adv`). 

```python
from a3t.DSL.transformation import Sub, SubChar, Del, Ins, InsChar, Swap

word_perturbation = [(Sub(True), 2), (Ins(), 2), (Del(), 2)]
char_perturbation = [(SubChar(True), 2), (InsChar(True), 2), (Del(), 2), (Swap(), 2)]
```

### Train


## Published Work

Yuhao Zhang, Aws Albarghouthi, Loris D’Antoni, Robustness to Programmable String Transformations via Augmented Abstract Training.

https://arxiv.org/abs/2002.09579
