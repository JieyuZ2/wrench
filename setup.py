from typing import Dict

from setuptools import find_packages, setup

# version.py defines the VERSION and VERSION_SHORT variables.
VERSION: Dict[str, str] = {}
with open("wrench/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)

# Use README.md as the long_description for the package
with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name="ws-benchmark",
    version=VERSION["VERSION"],
    author="Jieyu Zhang",
    author_email="jieyuzhang97@gmail.com",
    url="https://github.com/JieyuZ2/wrench",
    description="a benchmark for weak supervision",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="Apache License 2.0",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(exclude=("test*",)),
    include_package_data=True,
    install_requires=[
        'cytoolz>=0.11.0',
        'dill>=0.3.0,<0.4.0',
        'flyingsquid>=0.0.0a0',
        'future>=0.18.2',
        'higher>=0.2',
        'typing-extensions>=3.10.0.0',
        'torch>=1.2.0,<2.0.0',
        'torchvision>=0.10.0',
        'tqdm>=4.33.0,<5.0.0',
        'transformers>=4.6.1',
        'numpy>=1.16.5,<=1.22.3',
        'snorkel>=0.9.7',
        'seqeval>=1.2.2',
        'scikit-learn>=0.20.2,<0.25.0',
        'optuna>=2.7.0,<=2.10.0',
        'pandas>=1.1.5,<=2.0.0',
        'pillow>=8.3.2',
        'sentence-transformers>=1.1.1',
        'openml>=0.12.2',
        'cvxpy>=1.1.13,<=1.1.15',
        'scipy>=1.2.0,<2.0.0',
        'faiss-gpu>=1.7.1',
        'numbskull==0.1.1',
        'numba==0.43.0',
        'snorkel-metal>=0.5.0',
        'skweak>=0.2.13',
        'networkx>=2.2,<2.7',
        'hyperlm>=0.0.5',
    ],
    python_requires=">=3.6",
    keywords="machine-learning ai weak-supervision",
)
