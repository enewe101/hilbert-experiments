'''
Setup for the research code.  First make a virtualenvironment and activate it.
Then run this setup script.  Use Python2 :S
'''

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hilbert-research',
    version='0.0.1',
    description='Experiments with hilbert embedders',
    long_description=long_description,
    url='https://github.com/enewe101/hilbert-research',
    author='Edward Newell',
    author_email='edward.newell@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    keywords= (
        'word embeddings, NLP, word2vec, GloVe, FastText, skip-gram '
        'with negative sampling' 
    ),
    packages=['data_preparation', 'embedding_creation', 'shared', 'evaluation'],
	#indlude_package_data=True,
	package_data={'': ['README.md']},
	install_requires=[
        #'corenlpy', 
        'scipy', 'hilbert', 'nltk', 'sklearn-crfsuite', 'progress', 'nltk'
        'sklearn'
    ]
)
