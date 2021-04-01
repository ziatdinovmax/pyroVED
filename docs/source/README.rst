README
======

.. image:: https://github.com/ziatdinovmax/pyroVED/actions/workflows/actions.yml/badge.svg
    :target: https://github.com/ziatdinovmax/pyroVED/actions/workflows/actions.yml
    :alt: GiHub Actions
.. image:: https://badge.fury.io/py/pyroved.svg
        :target: https://badge.fury.io/py/pyroved
        :alt: PyPI version

pyroVED is an open-source package built on top of the Pyro probabilistic programming language for applications of variational encoder-decoder models in spectral and image analyses. The currently available models include variational autoencoders with translational and/or rotational invariance for unsupervised, class-conditioned, and semi-supervised learning, as well as *im2spec*-type models for predicting spectra from images and vice versa.
More models to come!

Installation
------------

Requirements
^^^^^^^^^^^^

*   python >= 3.6
*   pyro-ppl >= 1.6

Install pyroVED using pip:

.. code:: bash

   pip install pyroved

Latest (unstable) version
^^^^^^^^^^^^^^^^^^^^^^^^^

To upgrade to the latest (unstable) version, run

.. code:: bash

   pip install --upgrade git+https://github.com/ziatdinovmax/pyroved.git

Development
-----------

To run the unit tests, you'll need to have a pytest framework installed:

.. code:: bash

   python3 -m pip install pytest

Then run tests as:

.. code:: bash

   pytest tests

If this is your first time contributing to an open-source project, we highly recommend starting by familiarizing yourself with these very nice and detailed contribution `guidelines <https://github.com/firstcontributions/first-contributions>`_.