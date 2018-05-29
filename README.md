# GraphRNN: A Deep Generative Model for Graphs
This repository is a PyTorch implementation of GraphRNN, a graph generative model using auto-regressive model.

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)\*, [Rex Ying](https://cs.stanford.edu/people/rexy/)\*, [Xiang Ren](http://www-bcf.usc.edu/~xiangren/), [William L. Hamilton](https://stanford.edu/~wleif/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [GraphRNN: A Deep Generative Model for Graphs](https://arxiv.org/abs/1802.08773) (ICML 2018)

## Installation
Install PyTorch following the instuctions on the [official website](https://pytorch.org/). The code has been tested over PyTorch 0.2.0 and 0.4.0 versions.
```bash
conda install pytorch torchvision cuda90 -c pytorch
```
Then install the other dependencies.
```bash
pip install -r requirements.txt
```

## Test run
```bash
python main.py
```

## Code description
For the GraphRNN model:
`main.py` is the main executable file, and specific arguments are set in `args.py`.
`train.py` includes training iterations and calls `model.py` and `data.py`
`create_graphs.py` is where we prepare target graph datasets.

For baseline models:
B-A and E-R models are implemented in `baseline.py`
We implemented the DeepGMG paper based on the instruction of their paper in `main_DeepGMG.py`.


## Misc
Jesse Bettencourt and Harris Chan have made a great [slide](https://duvenaud.github.io/learn-discrete/slides/graphrnn.pdf) introducing GraphRNN in Prof. David Duvenaud’s seminar course [Learning Discrete Latent Structure](https://duvenaud.github.io/learn-discrete/).