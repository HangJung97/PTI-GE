<div align="center">

<div float="left" style="background-color: white;">
   <a href="https://www.creatis.insa-lyon.fr/site/en">
      <img src="https://www.creatis.insa-lyon.fr/site/sites/default/files/logo-creatis_0-1.png" width="25%" />
   </a>
   <a href="https://www.insa-lyon.fr">
      <img src="https://www.insa-lyon.fr/sites/www.insa-lyon.fr/files/logo-coul.png" width="25%" />
   </a>
</div>

# PyTorch Hands On <!-- omit in toc -->

[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/Pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![code-quality](https://github.com/creatis-myriad/ASCENT/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/creatis-myriad/ASCENT/actions/workflows/code-quality-main.yaml)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/HangJung97/PTI-GE/blob/main/LICENSE)

</div>

# Description <!-- omit in toc -->

This project serves as a starting point for anyone who wants to learn how to use PyTorch to perform
deep learning tasks, such as image classification, semantic segmentation, etc.

# Table of Contents <!-- omit in toc -->

- [Setup](#setup)
- [How to Run](#how-to-run)
- [Available Tutorials](#available-tutorials)
  - [Semantic Segmentation](#semantic-segmentation)
- [How to Contribute](#how-to-contribute)

# Setup

1. Download the repository:
   ```bash
   # Clone project
   git clone https://github.com/HangJung97/PTI-GE.git
   cd PTI-GE
   ```
2. Create a virtual environment (Conda is strongly recommended):
   ```bash
   # Create conda environment
   conda env create -f environment.yaml
   conda activate pti-ge
   ```
3. If you already have a python environment set aside for this project and just want to install the
   dependencies, you can do that using the following command:
   ```bash
   # Activate your environment: below is an example with conda
   conda activate <env name>
   # Install pytorch with conda or pip: below is an example with conda
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   # Install other dependencies
   pip install -e .
   ```

# How to Run

Once you've gone through the [setup](#setup) instructions above, you can start exploring the
tutorial's notebooks. We recommend using JupyterLab to run the notebooks, which can be launched by
running (from within your environment):

```bash
jupyter-lab
```

When you've launched JupyterLab's web interface, you can simply navigate to any of the
[tutorials listed below](#available-tutorials), and follow the instructions in there!

# Available Tutorials

## Semantic Segmentation

- [U-Net Applied to Echocardiography](notebooks/camus_segmentation.ipynb.ipynb)

# How to Contribute

If you want to contribute to the project, then you have to install development dependencies and
pre-commit hooks, on top of the basic setup for using the project, detailed [above](#setup). The
pre-commit hooks are there to ensure that any code committed to the repository meets the project's
format and quality standards.

```bash
# Install development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```
