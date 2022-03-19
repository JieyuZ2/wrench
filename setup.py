from typing import Dict

from setuptools import find_packages, setup

# version.py defines the VERSION and VERSION_SHORT variables.
# We use exec here so we don't import snorkel.
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
    description="a weak supervision learning benchmark",
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
        'cytoolz==0.11.0',
        'dill==0.3.4',
        'flyingsquid==0.0.0a0',
        'future>=0.18.2',
        'higher>=0.2.1',
        'typing-extensions>=3.10.0.0, <=3.10.0.2',
        'torch==1.9.0',
        'torchvision==0.10.0',
        'tqdm>=4.62.1',
        'transformers>=4.6.1,<=4.7.0',
        'numpy>=1.19.2,<=1.19.5',
        'snorkel==0.9.7',
        'seqeval==1.2.2',
        'scikit-learn==0.24.2',
        'optuna>=2.7.0,<=2.8.0',
        'pandas>=1.1.5',
        'pillow>=8.3.2',
        'sentence-transformers==1.1.1',
        'openml>=0.12.2',
        'cvxpy>=1.1.13,<=1.1.15',
        'scipy>=1.5.2,<=1.5.4',
        'faiss-gpu>=1.7.1',
        'numbskull==0.1.1',
        'numba==0.43.0',
        'snorkel-metal>=0.5.0',
        'spacy>=3.1.2,<=3.1.5',
        'skweak>=0.2.13',
        'networkx==2.7',
    ],
    python_requires=">=3.6",
    keywords="machine-learning ai weak-supervision",
)
