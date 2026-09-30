# Code Repository for "Isotropic Activation Functions Enable Deindividuated Neurons and Adaptive Topologies"

This repository contains all code for replication of all experiments within the paper.

Various experiments are structured as .ipynb files, whilst the .pkl normalisation files are needed to standardise the input data as described. The Dependencies directory contains the supporting functions used throughout the experiments.

## Experiment Zero
Times baseline training and neuroadaptive operations on CIFAR-10 to compare computational costs. There is one primary file
- "[Experiment 0] Timing Script.ipynb"

## Experiment One
Compares invariances before and after various neuroadaptations across CIFAR-10, CIFAR-100, and Caltech256 64x64. There are three primary files
- "[Experiment 1] Neuroadaptive Invariance (CIFAR10).ipynb"
- "[Experiment 1] Neuroadaptive Invariance (CIFAR100).ipynb"
- "[Experiment 1] Neuroadaptive Invariance (Caltech256-64x64).ipynb"

## Experiment Two
Tracks singular-value and bias evolution during training for isotropic and standard tanh networks. There are two primary files
- "[Experiment 2] Singular Value Evolutions (IsotropicTanh).ipynb"
- "[Experiment 2] Singular Value Evolutions (StandardTanh).ipynb"

## Experiment Three
Schedules the activation and deactivation of MNIST, Fashion-MNIST, and EMNIST A--J tasks during training to observe neuroadaptive changes. There is one primary file
- "[Experiment 3] Scheduled Task Neuroadaptation.ipynb"

## Experiment Four
Examines whether beginning with neural over- or under-abundance affects final accuracy when network widths are scheduled between fixed start and end widths. There are two primary files
- "[Experiment 4] Neuroabundance (CIFAR10).ipynb"
- "[Experiment 4] Neuroabundance (Caltech256).ipynb"

## Experiment Five
Examines the extent to which trained isotropic networks can be sparsified through alternating full diagonalisation whilst retaining their learned function. There are three primary files
- "[Experiment 5] Sparsification (CIFAR10).ipynb"
- "[Experiment 5] Sparsification (CIFAR100).ipynb"
- "[Experiment 5] Sparsification (Caltech256).ipynb"

## Experiment Six
Compares standard coordinatewise tanh MLPs against isotropic tanh MLPs across MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, Caltech256 64x64, and EMNIST A--J. There is one primary file
- "[Experiment 6] Cross-Dataset Tanh Comparisons.ipynb"

## Dependencies

Contains the `.py` files needed by the experiment notebooks, including the isotropic-network definitions, training utilities, GPU selection functions, and variable-dataset training code.
