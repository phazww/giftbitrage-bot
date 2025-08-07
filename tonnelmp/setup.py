from setuptools import setup
import pathlib

cwd = pathlib.Path(__file__).parent
long_description = (cwd / "README.md").read_text()

setup(
    name="tonnelmp",
    version="1.0.2",
    author="bleach",
    author_email="year0001@internet.ru",
    description="A Python Module for interacting with Tonnel Marketplace API",
    url="https://github.com/bleach-hub/tonnelmp",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["tonnelmp"],
    install_requires=[
    "curl_cffi>=0.5.7",
    "pycryptodome>=3.19.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)