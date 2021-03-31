# pyroVED

---
[![build](https://github.com/ziatdinovmax/pyroVED/actions/workflows/actions.yml/badge.svg)](https://github.com/ziatdinovmax/pyroVED/actions/workflows/actions.yml)
[![PyPI version](https://badge.fury.io/py/pyroved.svg)](https://badge.fury.io/py/pyroved)

pyroVED is an open-source pacakge built on top of the Pyro probabilistic programming language for applications of variational encoder-decoder models in spectral and image analysis. The currently available models include variational autoencoders with translational and/or rotational invariance for unsupervised, class-conditioned, and semi-supervised learning, as well as *im2spec*-type models for predicting spectra from images and vice versa.
More models to come!

## Examples
Please check out our interactive [examples](https://colab.research.google.com/github/ziatdinovmax/pyroVED/blob/main/examples/pyroVED_examples.ipynb) and let us know if you have any questions or if you would like to see the addition of any specific functionality!

## Installation

#### Requirements
- python >= 3.6
- pyro-ppl >= 1.6

Install pyroVED using pip:

```bash
pip install pyroved
```

#### Latest (unstable) version

To upgrade to the latest (unstable) version, run

```bash
pip install --upgrade git+https://github.com/ziatdinovmax/pyroved.git
```

## Development

To run the unit tests, you'll need to have a pytest framework:

```bash
python3 -m pip install pytest
```

Then run tests as:

```bash
pytest tests
```

If this is your first time contributing to an open-source project, we highly recommend to start with reading these very nice and detailed [guidelines](https://github.com/firstcontributions/first-contributions).
