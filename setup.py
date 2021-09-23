
from setuptools import find_packages, setup

version = "0.0.1"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="casia_itde_segmentation",
    version=version,
    description="casia_itde_segmentation",
    long_description=long_description,
    url="https://github.com/Gradiant/ai-dataset-casia_itde_segmentation",
    author="Gradiant",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
)