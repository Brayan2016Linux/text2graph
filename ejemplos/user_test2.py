#!/usr/bin/python3


import text2graph.data_graph as tg  #Para importar librería de grafos
import text2graph.tokenizer_text as tkt #Para importar tokenizador
import text2graph.ct_extractor as ce #Para importar crowdtangle
from text2graph.html_text import *

import time
import logging

import pandas as pd

if __name__ == '__main__':

    start_time = time.time()
    
    print('Inicio proceso:')
    print('Lectura archivo:')

    addr = "https://www.repretel.com/noticia/universitarios-protestan-nuevamente-en-asamblea-legislativa/"
 

    text3 = get_html_text(addr)
    print('Tiempo Lectura base datos: ' + str(time.time() - start_time) + 's')

    name_file="text1_"
    text = text3
    print(text)

    gap = 3
    
    
    print('Creación del grafo:' )

    print('Tokenize:')
    st1 = time.time()

    #Create Tokens from text:
    token_text = tkt.tokenize_text(text, with_stopwords=True)
    source, target = token_text.get_source_target_graph(gap=gap)
    
    #Create Graph:
    text_graph = tg(source, target)

    end1 = time.time()
    print('Tiempo total tokenizado: '+ str(end1 - st1))
    
    print('Graph:')
    st2 = time.time()
    g = text_graph.g
    #dfm = text_graph.metrics_csv('metrics.csv')

    #adjacency_matrix:
    #df2 = text_graph.adj_matrix_dataframe()
    #df2.to_csv('adjacency.csv', index=False)

    text_graph.plot_node_metric(metric='pagerank')

    print('Salvando gfxe:')
    #text_graph.save_to_gephi()

    print('Obteniendo layout')
    #text_graph.set_nx_layout(layout='spiral')

    print('Grafo con pagerank sin etiquetas:')

    text_graph.draw_graph_metrics(html=True, metric='degree', with_labels=True, first_n_values=6, with_values=False, cmap='Reds')

    end2 = time.time()
    print('Tiempo total analisis grafo: '+ str(end2 - st2))

    end_time = time.time()
    print('Tiempo total: '+ str(end_time - start_time))
    
    
