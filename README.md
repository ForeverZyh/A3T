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

## Get Started

We provide the training process of a word-level model and a char-level model on the SST2 dataset. 
Please see the `tests/test_run.py` for details.

### Prepare the Dataset and the Model

#### Dataset

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

#### Model

A3T targets at CNN models. `WordLevelSST2` and `CharLevelSST2` in `diffai.models` define two CNN models that were used in the experiments of this paper.
Remark that some literature point out that adding a linear layer between embedding layer and the first layer yields better results. 
This phenomenon also appears in CNN models. Thus, for better results, we recommend adding a linear layer by calling `diffai.componenets.Linear(out_dim)`.

### Customize the String Transformations

In general, A3T supports customized string transformations provided by the users.
The `DSL.transformation` contains several string transformations already defined and used in the experiments of the paper, namely,
`Sub`, `SubChar`, `Del`, `Ins`, `InsChar`, and `Swap`. Among those transformations, `Sub`, `SubChar`, and `Swap` are labeled as length-preserving transformations, which allows robust training.

One can define their own string transformations by implementing the abstract class `Transformation` and two functions `get_pos` and `transformer` as described in our paper. 
`get_pos` accepts a list of input tokens and returns a list of position pairs (start, end).
`transformer` accepts a list of input tokens and a start-end position pair and returns an iterator which enumerates the possible transformations at the start-end position.

```python
from abc import ABC, abstractmethod

class Transformation(ABC):
    def __init__(self, length_preserving=False):
        """
        A default init function.
        """
        super().__init__()
        self.length_preserving = length_preserving

    @abstractmethod
    def get_pos(self, ipt):
        # get matched positions in input
        pass

    @abstractmethod
    def transformer(self, ipt, start_pos, end_pos):
        # transformer for a segment of input
        pass

    def sub_transformer(self, ipt, start_pos, end_pos):
        # substring transformer for length preserving transformation
        assert self.length_preserving
```
Notice that we need to implement `sub_transformer` if the transformation is a length-preserving transformation. 

#### Define a perturbation space

A perturbation space is in the form of `[(Trans_1, delta_1), ..., (Trans_n, delta_n)]`. Ideally, the perturbation is a set of the string transformations, but we use a list to store the perturbation space. 
In other words, we impose an order in the perturbation space, which will effect the HotFlip attack (see TODO in `GeneralHotFlipAttack.gen_adv`). 

```python
from a3t.DSL.transformation import Sub, SubChar, Del, Ins, InsChar, Swap

word_perturbation = [(Sub(True), 2), (Ins(), 2), (Del(), 2)]
char_perturbation = [(SubChar(True), 2), (InsChar(True), 2), (Del(), 2), (Swap(), 2)]
```

### Train

The main training pipeline is in `diffai.train.train`

```python
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
```
The `args` argument contains various training hyper-parameters, see `tests/test_run.Args` for a default version of hyper-parameter settings.

In all, `tests/test_run` contains a complete process of training a `SST2WordLevel` model and a `SST2CharLevel` model on a fragment of dataset.

## Published Work

Yuhao Zhang, Aws Albarghouthi, Loris D’Antoni, Robustness to Programmable String Transformations via Augmented Abstract Training.

https://arxiv.org/abs/2002.09579
