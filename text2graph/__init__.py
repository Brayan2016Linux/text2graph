"""
Analizador de texto mediante grafos, utiliza los textos y crea grafos para ser visualizados
por networkx, graphtools o bien por gephi al exportar la matriz de adyacencia.

"""

import text2graph

from text2graph.tokenizer_text import *
from text2graph.data_graph import *
from text2graph.ct_extractor import *
from text2graph.html_text import *
