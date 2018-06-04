# GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model
This repository is the official PyTorch implementation of GraphRNN, a graph generative model using auto-regressive model.

[Jiaxuan You](https://cs.stanford.edu/~jiaxuan/)\*, [Rex Ying](https://cs.stanford.edu/people/rexy/)\*, [Xiang Ren](http://www-bcf.usc.edu/~xiangren/), [William L. Hamilton](https://stanford.edu/~wleif/), [Jure Leskovec](https://cs.stanford.edu/people/jure/index.html), [GraphRNN: Generating Realistic Graphs with Deep Auto-regressive Model](https://arxiv.org/abs/1802.08773) (ICML 2018)

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
* B-A and E-R models are implemented in `baselines/baseline_simple.py`.
* [Kronecker graph model](https://cs.stanford.edu/~jure/pubs/kronecker-jmlr10.pdf) is implemented in the SNAP software, which can be found in `https://github.com/snap-stanford/snap/tree/master/examples/krongen` (for generating Kronecker graphs), and `https://github.com/snap-stanford/snap/tree/master/examples/kronfit` (for learning parameters for the model).
* MMSB is implemented using the EDWARD library (http://edwardlib.org/), and is located in
  `baselines`.
* We implemented the DeepGMG model based on the instructions of their [paper](https://arxiv.org/abs/1803.03324) in `main_DeepGMG.py`.
* We implemented the GraphVAE model based on the instructions of their [paper](https://arxiv.org/abs/1802.03480) in `baselines/graphvae`.

Parameter setting:
To adjust the hyper-parameter and input arguments to the model, modify the fields of `args.py`
accordingly.
For example, `args.cuda` controls which GPU is used to train the model, and `args.graph_type`
specifies which dataset is used to train the generative model. See the documentation in `args.py`
for more detailed descriptions of all fields.

## Outputs
There are several different types of outputs, each saved into a different directory under a path prefix. The path prefix is set at `args.dir_input`. Suppose that this field is set to `./`:
* `./graphs` contains the pickle files of training, test and generated graphs. Each contains a list
  of networkx object.
* `./eval_results` contains the evaluation of MMD scores in txt format.
* `./model_save` stores the model checkpoints
* `./nll` saves the log-likelihood for generated graphs as sequences.
* `./figures` is used to save visualizations (see Visualization of graphs section).

## Evaluation
The evaluation is done in `evaluate.py`, where user can choose which settings to evaluate.
To evaluate how close the generated graphs are to the ground truth set, we use MMD (maximum mean discrepancy) to calculate the divergence between two _sets of distributions_ related to
the ground truth and generated graphs.
Three types of distributions are chosen: degree distribution, clustering coefficient distribution.
Both of which are implemented in `eval/stats.py`, using multiprocessing python
module. One can easily extend the evaluation to compute MMD for other distribution of graphs.

We also compute the orbit counts for each graph, represented as a high-dimensional data point. We then compute the MMD
between the two _sets of sampled points_ using ORCA (see http://www.biolab.si/supp/orca/orca.html) at `eval/orca`. 
One first needs to compile ORCA by 
```bash
g++ -O2 -std=c++11 -o orca orca.cpp` 
```
in directory `eval/orca`.
(the binary file already in repo works in Ubuntu). 

To evaluate, run 
```bash
python evaluate.py
```
Arguments specific to evaluation is specified in class
`evaluate.Args_evaluate`. Note that the field `Args_evaluate.dataset_name_all` must only contain
datasets that are already trained, by setting args.graph_type to each of the datasets and running
`python main.py`.

## Visualization of graphs
The training, testing and generated graphs are saved at 'graphs/'.
One can visualize the generated graph using the function `utils.load_graph_list`, which loads the
list of graphs from the pickle file, and `util.draw_graph_list`, which plots the graph using
networkx. 


## Misc
Jesse Bettencourt and Harris Chan have made a great [slide](https://duvenaud.github.io/learn-discrete/slides/graphrnn.pdf) introducing GraphRNN in Prof. David Duvenaud’s seminar course [Learning Discrete Latent Structure](https://duvenaud.github.io/learn-discrete/).

