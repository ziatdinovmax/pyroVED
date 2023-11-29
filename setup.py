__author__ = "Maxim Ziatdinov"
__copyright__ = "Copyright Maxim Ziatdinov (2021)"
__maintainer__ = "Maxim Ziatdinov"
__email__ = "maxim.ziatdinov@ai4microcopy.com"

from setuptools import setup, find_packages
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(module_dir, 'pyroved/__version__.py')) as f:
    __version__ = f.read().split("'")[1]

if __name__ == "__main__":
    setup(
        name='pyroved',
        python_requires='>=3.6',
        version=__version__,
        description='Variational encoder-decoder models in Pyro probabilistic programming language',
        long_description=open(os.path.join(module_dir, 'README.md')).read(),
        long_description_content_type='text/markdown',
        url='https://github.com/ziatdinovmax/pyroVED/',
        author='Maxim Ziatdinov',
        author_email='maxim.ziatdinov@ai4microcopy.com',
        license='MIT license',
        packages=find_packages(),
        zip_safe=False,
        install_requires=[
            'pyro-ppl>=1.6.0',
            'matplotlib>=3.2',
            'torchvision>=0.7.0'
        ],
        classifiers=['Programming Language :: Python',
                     'Development Status :: 3 - Alpha',
                     'Intended Audience :: Science/Research',
                     'Operating System :: OS Independent',
                     'Topic :: Scientific/Engineering']
    )
