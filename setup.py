#!/usr/bin/python3
# -*- coding: utf-8 -*-

#Install by setuptools or distutils:
try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

#Long description:
import pathlib
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

#Requirements:
requirements = ["networkx>=2.4", 
                "pandas>=0.24.1",
                "pandas-datareader>=0.7.0", 
                "plotly>=3.4.1",
                "scipy>=1.3.0",
                "numpy>=1.16.4",
                "pyreadr>=0.2.9",
                "spacy>=2.3.2",
                "tldextract>=2.2.3",
                "nltk>=3.4.0",
                "mpld3>=0.5.2",
                "beautifulsoup4>=4.9.1",
                "requests>=2.25.0",
                "tldextract>=2.2.3",
                "lxml>=4.3.1",
                "google>=3.0.0",
                "matplotlib==3.3.2",
                "python-louvain>=0.13"       
]

#Setup configuration:
setup(
    name="text2graph", #Tentativo
    version="0.0.2",
    description="Python package to generate graph from a text with networkx",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Brayan Rodríguez",
    author_email="bradrd2009jp@gmail.com",
    license="",
    url="",
    classifiers =[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Python expert users",
        "License ::˝OSI Approved :: MIT", #Tentativo
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Operating System :: OS Independent'
    ],
    keywords="graph data visualizer text statistics analyzer",
    install_requires=requirements,
    packages = ['text2graph'],
    python_requires="~=3.0",
    #setup_requires = ['flake8']
)

from install_model import *
model = "es_core_news_md"
#install_model(model)
