from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md")) as f:
    long_description = f.read()

setup(
    name="featuretools_sklearn_transformer",
    version="1.0.0",
    author="Feature Labs, Inc.",
    author_email="support@featurelabs.com",
    license="BSD 3-clause",
    url="http://www.featurelabs.com/",
    python_requires=">=3.7, <4",
    install_requires=open("requirements.txt").readlines(),
    packages=find_packages(),
    description="Featuretools Transformer for Scikit-Learn Pipeline use.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    entry_points={
        "featuretools_plugin": [
            "wrappers = featuretools_sklearn_transformer",
        ],
    },
)
