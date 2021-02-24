text2graph. (Nombre pendiente de cambio)

text2graph is an small Python package that can be use to obtain the visualization of graph given an .csv file with information about nodes and its economical activity to find posible relations and painted according with specified filters.

## Installation

To install this package, you need open project and write next command:

'''
$python3 setup.py install --user
'''

If you are installing this package in WindowsOS you should use next, if previous command had not worked:

'''
$python setup.py install --user
'''

## Before use:

May be you should to install any of models of spacy, this is the command:
'''
python -m spacy download es_core_news_sm
'''

or

'''
python3 -m spacy download es_core_news_sm
'''

Consulting:
https://spacy.io/usage/ to get more model

##Install model:
You can install model uncommenting next line in setup.py file:

'''
install_model(<name_model>)
'''
