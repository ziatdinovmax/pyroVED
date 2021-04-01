# pyroVED

---
[![build](https://github.com/ziatdinovmax/pyroVED/actions/workflows/actions.yml/badge.svg)](https://github.com/ziatdinovmax/pyroVED/actions/workflows/actions.yml)
[![Documentation Status](https://readthedocs.org/projects/pyroved/badge/?version=latest)](https://pyroved.readthedocs.io/en/latest/README.html)
[![PyPI version](https://badge.fury.io/py/pyroved.svg)](https://badge.fury.io/py/pyroved)

pyroVED is an open-source package built on top of the Pyro probabilistic programming language for applications of variational encoder-decoder models in spectral and image analyses. The currently available models include variational autoencoders with translational and/or rotational invariance for unsupervised, class-conditioned, and semi-supervised learning, as well as *im2spec*-type models for predicting spectra from images and vice versa.
More models to come!

## Examples
The easiest way to start using pyroVED is via [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb), which is a free research tool from Google for machine learning education and research built on top of Jupyter Notebook. The following notebooks can be executed in Google Colab by simply clicking on the "Open in Colab" icon:

*   Shift-VAE: Mastering the 1D shifts in spectral data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/pyroVED/blob/master/examples/shiftVAE.ipynb)

*   r-VAE: Disentangling image content from rotations [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/pyroVED/blob/master/examples/rVAE.ipynb)

*   j(r)-VAE: Learning (jointly) discrete and continuous representations of data [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/pyroVED/blob/main/examples/jrVAE.ipynb)

*   ss(r)-VAE: Semi-supervised learning from data with orientational disorder [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ziatdinovmax/pyroVED/blob/main/examples/ssrVAE.ipynb)

## Installation

#### Requirements
*   python >= 3.6
*   pyro-ppl >= 1.6

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

To run the unit tests, you'll need to have a pytest framework installed:

```bash
python3 -m pip install pytest
```

Then run tests as:

```bash
pytest tests
```

If this is your first time contributing to an open-source project, we highly recommend starting by familiarizing yourself with these very nice and detailed contribution [guidelines](https://github.com/firstcontributions/first-contributions).
