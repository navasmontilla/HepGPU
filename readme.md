# HepGPU:  A Python package for the simulation of virus spread and inmune response in the liver


## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation-instructions)
3. [Automated test](#automated-test)


## Introduction

`HepGPU` is a Python package for the simulation of virus spread and inmune response in the liver.

## Installation instructions

Install Python 3 and create virtual environment:

```
sudo apt update
sudo apt install -y python3 python3-pip
python3 -m venv myenv
source myenv/bin/activate
```

Install HepGPU:

```
pip install git+https://github.com/navasmontilla/HepGPU.git
```


Install dependencies:

```
pip3 install matplotlib numpy zarr matplotlib 
```

## Automated test

All demos can be run on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./examples/)
