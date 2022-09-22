#!/usr/bin/env python
import os
from importlib.util import module_from_spec, spec_from_file_location
from typing import List

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)


def _load_py_module(fname, pkg="pytorch_gleam"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_ROOT, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _load_readme_description(path_dir: str, homepage: str) -> str:
    path_readme = os.path.join(path_dir, "README.md")
    with open(path_readme, encoding="utf-8") as f:
        text = f.read()

    github_source_url = os.path.join(homepage, "blob/master")
    # replace relative repository path to absolute link to the release
    text = text.replace("docs/images/", f"{os.path.join(github_source_url, 'docs/images/')}")

    return text


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    requirements = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http") or "@http" in ln:
            continue
        if ln:
            requirements.append(ln)
    return requirements


about = _load_py_module("__about__.py")


# python setup.py sdist
# twine upload .\dist\pytorch-gleam-VERSION.tar.gz
long_description = _load_readme_description(_PATH_ROOT, homepage=about.__homepage__)


# Setting up
setup(
    name="pytorch-gleam",
    version=about.__version__,
    author=about.__author__,
    author_email=about.__author_email__,
    description=about.__docs__,
    url=about.__homepage__,
    license=about.__license__,
    long_description=long_description,
    packages=find_packages(exclude=["tests*", "pg_examples*", "docs*"]),
    long_description_content_type="text/markdown",
    zip_safe=False,
    include_package_data=True,
    keywords=[
        "social media",
        "twitter",
        "pytorch",
        "torch",
        "pytorch_lightning",
        "nlp",
        "deep learning",
    ],
    python_requires=">=3.6",
    setup_requires=[],
    install_requires=_load_requirements(_PATH_ROOT),
    project_urls={
        "Bug Tracker": "https://github.com/Supermaxman/pytorch-gleam/issues",
        "Source Code": "https://github.com/Supermaxman/pytorch-gleam",
    },
    download_url="https://github.com/Supermaxman/pytorch-gleam",
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    entry_points={
        "console_scripts": [
            "free-gpus=pytorch_gleam.gpu.free_gpus:main",
            "request-gpus=pytorch_gleam.gpu.request_gpus:main",
            "ex-queue=pytorch_gleam.exqueue.exqueue:main",
            "ex-bqueue=pytorch_gleam.exqueue.exbqueue:main",
            "ex-start=pytorch_gleam.exqueue.exstart:main",
            "ex-stat=pytorch_gleam.exqueue.exstat:main",
            # TODO kill default queue and kill running processes
            # "ex-end=pytorch_gleam.exqueue.exend:main",
            "ex-copy=pytorch_gleam.ex.copy:main",
            "ex-rm=pytorch_gleam.exqueue.exrm:main",
            "t-parse=pytorch_gleam.parse.tparse:main",
            "f-parse=pytorch_gleam.parse.fparse:main",
            "e-parse=pytorch_gleam.parse.eparse:main",
            "gleam-search-tweet-index=pytorch_gleam.search.tweet_index_search:main",
            "gleam-rerank=pytorch_gleam.search.rerank:main",
            "gleam-rerank-format=pytorch_gleam.search.rerank_format:main",
            "gleam-tweet-to-jsonl=pytorch_gleam.search.tweet_to_jsonl:main",
            "gleam-search-candidates=pytorch_gleam.search.select_candidates:main",
            "gleam-train=pytorch_gleam.ex.train:main",
            "gleam-test=pytorch_gleam.ex.test:main",
            "gleam-validate=pytorch_gleam.ex.validate:main",
            "gleam-predict=pytorch_gleam.ex.predict:main",
            "gleam=pytorch_gleam.ex.gleam:main",
        ],
    },
)
