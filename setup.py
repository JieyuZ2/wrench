from setuptools import find_packages, setup

with open("requirements.txt") as f:
    REQUIREMENTS = f.read().splitlines()

# read the contents of abstract file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()


def setup_package() -> None:
    setup(
        name="wrench",
        version="1.0.0",
        url="https://github.com/Jieyuz2/wrench",
        author="Wrench authors",
        long_description=long_description,
        python_requires=">=3.6",
        packages=find_packages(),
        install_requires=REQUIREMENTS,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    )


if __name__ == "__main__":
    setup_package()
