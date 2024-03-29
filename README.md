<div align="center">

# Accelerating equilibrium spin-glass simulations using quantum data and deep learning

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-SciPost%20Phys.%2015,%20018%20(2023)-B31B1B.svg)](https://scipost.org/10.21468/SciPostPhys.15.1.018)
[![DOI](https://zenodo.org/badge/542095061.svg)](https://zenodo.org/badge/latestdoi/542095061)

</div>

## Description

Sampling from the low-temperature Boltzmann distribution of spin glasses is a hard computational task, relevant for physics research and important optimization problems in engineering and finance.
Adiabatic quantum computers are being used to tackle the optimization task, corresponding to find the lowest energy spin configuration. In this paper we show how to exploit quantum annealers to accelerate equilibrium Markov chain Monte Carlo simulations of spin glasses at low but finite temperature. Generative neural networks are trained on spin configurations produced by the D-Wave quantum annealers. Moreover, they are used to generate smart proposals for the Metropolis-
Hastings algorithm. In particular, we explore hybrid schemes by combining neural and single spin-flip proposals, as well as D-Wave and classical Monte Carlo training data. The hybrid algorithm
outperforms the single spin-flip Metropolis-Hastings algorithm and it is competitive with parallel tempering in terms of correlation times, with the significant benefit of a faster equilibration.

For a visual summary (with some results) you can have a look to the notebook [`article_figures`](article_figures.ipynb) without re-running anything. If you want to reproduce the same plots of the article, dowload the data and install the dependecies before run it.    

## How to run
Install dependencies
```yaml
# clone project
git clone https://github.com/gscriva/n-mcmc
cd n-mcmc

# [OPTIONAL] create conda environment
bash bash/setup_conda.sh

# install requirements
pip install -r requirements.txt
```
Get the data from the Zenodo directory [10.5281/zenodo.7250436](https://doi.org/10.5281/zenodo.7250436) and move them in [data/](data/), the directory must be organized as follow:
```
data
  ├── couplings
  |     ├── 100.txt
  |     .
  |     .
  |     └── 484-z8.txt
  ├── data_for_fig
  |     ├── data_fig1.csv
  |     .
  |     .
  |     .
  |     └── data_fig7.csv
  └── datasets
        ├── 100-1mus
        |    ├── train_1us.npy
        |    └── train_1us.npy
        .
        .
        .
        └── 484-z8-1mus
            ├── ...
            └── ...
```

Train model with default configuration
```yaml
# default
python run.py
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
# model with 100 spins
python run.py experiment=100spin-1nn.yaml

# models with 484 spins
python run.py experiment=484spin-3nn.yaml
```

You can generate from the trained model with 
```yaml
python predict.py --ckpt-path=logs/the/trained/model/path.ckpt --model=made 
```

<br>

## Citation   
```
@article{10.21468/SciPostPhys.15.1.018,
	title={{Accelerating equilibrium spin-glass simulations using quantum annealers  via generative deep learning}},
	author={Giuseppe Scriva and Emanuele Costa and Benjamin McNaughton and Sebastiano Pilati},
	journal={SciPost Phys.},
	volume={15},
	pages={018},
	year={2023},
	publisher={SciPost},
	doi={10.21468/SciPostPhys.15.1.018},
	url={https://scipost.org/10.21468/SciPostPhys.15.1.018},
}
```   
