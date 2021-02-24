#!/usr/bin/python3
# -*- coding: utf-8 -*-

# =============================================================================
# Graph Analyzer Graph Utils
# =============================================================================
#
# Miscellaneous utility functions to be used in graph analysis.
# @Author: Brayan Rodriguez <bradrd2009jp@gmail.com>
# @Organization: LIIT-UNED 2020
# @Version: 0.0.4

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#Librerias Externas:
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib as mpl
import matplotlib.pyplot as plt

from networkx.algorithms import community as comm
from networkx.algorithms.structuralholes import constraint, local_constraint, effective_size
from community import community_louvain as cm #requiered: python-louvain

import mpld3

from datetime import date
import itertools


__all__ = ['data_graph']

# =========================== CLASS DATA GRAPH =============================================

class data_graph():
    def __init__(self, source, target, is_directed=False, is_multi=False):
        """Crea un grafo a partir de dos listas source y target, y parámetros de tipo de grafo"""
        self.source = [str(i) for i in source]
        self.target = [str(i) for i in target]
        self.is_directed = is_directed
        self.is_multi = is_multi
        self.pos = None
        self.vertex_labels = list(set(self.source + self.target))
        if len(source)==len(target):
            self.g = self.create_graph()
        if self.pos == None:
            self.pos = self.set_nx_layout(layout='spring')
        self.backup_g = self.g
        self.metrics_dict = {} #For memoizing data

        self.__version___="0.0.4"

    def create_graph(self):
        """create_graph: crea el grafo agregando las aristas según el archivo
           de etiquetas, que se crea con la lista de source y target.

           Retorna: grafo de networkx
        """
        is_directed = self.is_directed
        is_multi = self.is_multi
        if not is_directed and not is_multi:
            g = nx.Graph()
        elif is_directed and not is_multi:
            g = nx.DiGraph()
        elif not is_directed and is_multi:
            g = nx.MultiGraph()
        else:
            g = nx.MultiDiGraph()
        #Add arist to graph:
        vertex_labels = self.vertex_labels
        src = self.source
        trg = self.target
        for i in range(len(src)):
            if g.has_edge(vertex_labels.index(src[i]), vertex_labels.index(trg[i])):
                g[vertex_labels.index(src[i])][vertex_labels.index(trg[i])]['weight'] += 1
            else:
                g.add_edge(vertex_labels.index(src[i]), vertex_labels.index(trg[i]), weight = 1)
        return g

# ===================== MISCELANEA ========================================================

    def add_label_attrib(self):
        """Agrega el atributo de etiquetas a los nodos"""
        labels = self.vertex_labels
        label_dict = {}
        for i in self.g.nodes:
            label_dict[i]=labels[i]
        nx.set_node_attributes(self.g, label_dict, "labels")

    def set_node_attributes(self, attr_dict, name):
        """Agrega el atributo a partir de un diccionario de nodos y el nombre del atributo"""
        nx.set_node_attributes(self.g, attr_dict, name)

    def get_node_attributes(self, attribute):
        """Obtiene el atributo de los nodos a partir del nombre, devuelve un diccionario"""
        attribute = nx.get_node_attributes(self.g, attribute)
        return attribute
    
    def list_nodes_attributes(self):
        """Devuelve la lista de atributos en el grafo actual"""
        g = self.g
        list_attrib = list(set([k for n in g.nodes for k in g.nodes[n].keys()]))
        return list_attrib
   
    def get_dict_node_attrib(self):
        """Devuelve el diccionario de atributo"""
        list_nodes_attributes = self.list_nodes_attributes()
        dict_node_attr = {}
        g = self.g
        for i in list_nodes_attributes:
            attr = nx.get_node_attributes(g, i)
            dict_node_attr[i] = attr
        return dict_node_attr


    def save_edge_list(self, save_as='edges.csv', data=True):
        """Exporta la lista de aristas para ser utilizados por otros programas de analisis de redes"""
        g = self.g
        output = "Source,Target,Weight\n"
        edge_list = [i.split(',') for i in nx.generate_edgelist(g, delimiter=',', data=['weight'])]
        for i in edge_list:
            output += "%s,%s,%s\n"%(self.vertex_labels[int(i[0])], self.vertex_labels[int(i[1])], i[2])
        output_name = save_as
        with open(output_name, 'w') as file:
            file.write(output)
            file.close

    def graph(self):
        """Devuelve el grafo como objeto de networkx, equivalente a: .g"""
        return self.g

    def labels(self):
        """Crea el diccionario de etiquetas, para uso interno del programa"""
        return {i : self.vertex_labels[i] for i in range(0, len(self.vertex_labels))}

    def self_loop_nodes_number(self):
        """Devuelve el número de nodos con lazos a sí mismo"""
        return nx.number_of_selfloops(self.g) 

    def remove_node(self, node_number):
        """Remueve el nodo del grafo conociendo el número de nodo"""
        self.g.remove_node(node_number)
    
    def remove_node_by_label(self, label):
        """Remueve el nodo del grafo conociendo la etiqueta"""
        if label in self.vertex_labels:
            self.g.remove_node(self.vertex_labels.index(label))
        else:
            print("Label %s not found, try again."%label)

    def reset(self):
        """Restaura el grafo al original guardado en una copia de seguridad"""
        self.g = self.backup_g


# ==================  NODE FREQUECY ====================================================

    def get_node_frequency_df(self, save = False, save_as='node_freq.csv', index=False):
        """Obtiene el dataframe con la frecuencia de nodos en las listas de source y target"""
        nodes = self.source + self.target
        labels = list(set(nodes))
        freq = list()
        df = pd.DataFrame()
        for i in labels:
            freq.append(nodes.count(i))
        df['labels'] = labels
        df['frequency'] = freq
        if save:
            df.to_csv(save_as, index=index)
        return df
    
    def plot_node_frequency(self, tail_number=25, ascending=True, save=True, save_as='node_freq.png', html=False, html_name='node_freq.html', figw = 10, figh = 10, color='g'):
        """Grafica la frecuencia de los nodos, con 25 por default aparecerán en el top
           save, es booleana y guardará el archivo con el nombre designado en save_as, 
           de lo contrario mostrará en pantalla.
        """
        plt.close('all')
        df = self.get_node_frequency_df()
        df = df.sort_values('frequency',ascending=ascending).tail(tail_number)
        fig, ax = plt.subplots(1,1)
        df.plot(kind='barh', ax = ax, color=color)
        ax.set_yticklabels(df['labels'].tolist())
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Node')
        ax.figure.set_size_inches(figw, figh)
        if save or html:
            if save:
                plt.savefig(save_as)
            if html:
                mpld3.save_html(plt.gcf(), html_name)
        else:
            plt.show()


# ================== ADJACENCY MATRIX ==================================================

    def get_adj_matrix(self):
        """Devuelve la matriz de adjacencia del grafo, para trabajo con numpy"""
        A = nx.adjacency_matrix(self.g).todense()
        return A

    def adj_matrix_dataframe(self):
        """Devuelve la matriz de adjacencia en dataframe"""
        g = self.g
        columns = [self.vertex_labels[i] for i in g.nodes]
        A = self.get_adj_matrix()
        df = pd.DataFrame(A, columns = columns)
        #Add name of columns
        df[''] = np.transpose(columns)
        #Reorganize Columns
        cols = [''] + [col for col in df if not col == '']
        df = df[cols]
        return df

    #V.0.0.4
    def adj_matrix_csv(self, save_as='adjacency.csv', index=False):
        """Devuelve la matriz de adjacencia del grafo y la guarda en .csv"""
        df = self.adj_matrix_dataframe()
        df.to_csv(save_as, index=index)

    #V.0.0.4
    def get_inc_matrix(self, nodelist=None, edgelist=None, oriented=False, weight=None):
        """Devuelve la matriz de adjacencia del grafo, para trabajo con numpy"""
        A = nx.incidence_matrix(self.g, nodelist=nodelist, edgelist=edgelist, oriented=oriented, weight=weight).todense()
        return A

    #V.0.0.4
    def incidence_matrix_dataframe(self, nodelist=None, edgelist=None, oriented=False, weight=None):
        """Devuelve la matriz de adjacencia en dataframe"""
        g = self.g
        rows = [self.vertex_labels[i] for i in g.nodes]
        A = self.get_inc_matrix(nodelist=nodelist, edgelist=edgelist, oriented=oriented, weight=weight)
        columns = ['E'+str(i+1) for i in range(len(np.matrix(A.T)))]
        df = pd.DataFrame(A, columns = columns)
        #Add name of columns
        df[''] = rows
        #Reorganize Columns
        cols = [''] + [col for col in df if not col == '']
        df = df[cols]
        return df

    #V.0.0.4
    def incidence_matrix_csv(self, save_as='incidence.csv', index=False, **kwargs):
        """Devuelve la matriz de incidencia del grafo y la guarda en .csv"""
        df = self.incidence_matrix_dataframe(**kwargs)
        df.to_csv(save_as, index=index)

    def number_of_vertex(self):
        """Devuelve el número de vértices"""
        return self.g.number_of_nodes()
    
    def number_of_edges(self):
        """Devuelve el número de aristas"""
        return self.g.number_of_edges()
      
    def shortest_distance_matrix(self):
        """Devuelve la distancia más corta"""
        g = self.g
        dist = dict(nx.all_pairs_shortest_path_length(self.g))
        distance = list()
        for i in g.nodes:
            temp = list()
            keys = [k for k in dist[i].keys()]
            vals = [v for v in dist[i].values()]
            #print(keys)
            #print(vals)
            for j in g.nodes:
                if j in keys:
                    temp.append(vals[keys.index(j)])
                else:
                    temp.append(0)
            distance.append(temp)
        return distance
    
    def shortest_distance_dataframe(self):
        """Devuelve el dataframe con la distancia más corta para cada nodo"""
        #Matrix distance:
        g = self.g
        columns = [self.vertex_labels[i] for i in g.nodes] 
        dist = dict(nx.all_pairs_shortest_path_length(self.g))
        distance = list()
        for i in g.nodes:
            temp = list()
            keys = [k for k in dist[i].keys()]
            vals = [v for v in dist[i].values()]
            #print(keys)
            #print(vals)
            for j in g.nodes:
                if j in keys:
                    temp.append(vals[keys.index(j)])
                else:
                    temp.append(0)
            distance.append(temp)

        df = pd.DataFrame(distance, columns = columns)
        #Add name of columns
        df['Variables'] = np.transpose(columns)
        #Reorganize Columns
        cols = ['Variables'] + [col for col in df if not col == 'Variables']
        df = df[cols]
        return df

    def logging_message(self, message):
        """Imprime un mensaje, uso interno del programa"""
        print(message)
   
    def empty_dict(self):
        """Devuelve un diccionario vacío, cuando no es posible realizar el cálculo de una métrica"""
        dict_ = {}
        for i in self.g.nodes():
              dict_[i] = np.nan
        return dict_
  
    #Change graph to undirected to calculate some of metrics
    def get_undirected(self):
        """Devuelve una versión del gráfico en no dirigido para realizar algunos cálculos"""
        g = self.g
        return g.to_undirected()

    #====================== Matrix Visualization =============================================

    #V.0.0.4
    def adjacency_matrix_plot(self, display=True, save_as="adjacency_matrix.png", with_sticks = True, fontsize=7, cmap='YlOrRd'):
        """Dibuja la matriz de adjacencia del grafo"""
        g = self.g
        matrix = self.get_adj_matrix()
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)

        if with_sticks:
            labels = [self.vertex_labels[i] for i in g.nodes]
            order = [labels.index(i) for i in labels]
            plt.xticks(order, labels, rotation=90, fontsize = fontsize)
            plt.yticks(order, labels, fontsize = fontsize)

        if display:
            plt.show()
        else:
            if self.return_extension(save_as) == 'png':
                output_name = save_as
            else:
                output_name = save_as + '.png'
                plt.savefig(output_name)

#=====================================METRICS ========================================
#DEGREES:  
    def indegree(self):
        """Devuelve un diccionario con el valor indegree para cada nodo, si el grafo es dirigido"""
        g = self.g
        if nx.is_directed(g):
            dict_ = {}
            for i in g.nodes():
                dict_[i] = g.in_degree(i)
            return dict_
        else:
            dict_ = {}
            for i in g.nodes():
                dict_[i] = 0
            return dict_
        self.metrics_dict['indegree'] = dict_
    
    def outdegree(self):
        """Devuelve un diccionario con el valor outdegree para cada nodo, si el grafo es dirigido"""
        g = self.g
        if nx.is_directed(g):
            dict_ = {}
            for i in g.nodes():
                dict_[i] = g.out_degree(i)
            return dict_
        else:
            dict_ = {}
            for i in g.nodes():
                dict_[i] = 0
            return dict_
        self.metrics_dict['outdegree'] = dict_
    
    def degree(self):
        """Devuelve un diccionario con el valor degree para cada nodo"""
        g = self.g
        dict_ = {}
        for i in g.nodes():
            dict_[i] = g.degree(i)
        return dict_
        self.metrics_dict['degree'] = dict_
    

#CENTRALITY:
    def eccentricity(self):
        """Devuelve un diccionario con el valor eccentricity para cada nodo, si el grafo es conectado"""
        try:
            g = self.g
            self.metrics_dict['eccentricity'] = nx.eccentricity(g)
            return self.metrics_dict['eccentricity']
        except nx.exception.NetworkXError:
            self.logging_message("Excentricity: Graph is not connected.")
            return self.empty_dict()

    def harmonic_centrality(self):
        """Devuelve un diccionario con el valor harmonic_centrality para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['harmonic'] = nx.harmonic_centrality(g)
            return self.metrics_dict['harmonic']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Harmonic: Power iteration failed.")
            return self.empty_dict()

    def closeness_centrality(self):
        """Devuelve un diccionario con el valor closeness_centrality para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['closeness'] = nx.closeness_centrality(g)
            return self.metrics_dict['closeness']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Closeness: Power iteration failed.")
            return self.empty_dict() 
            
    def eigenvector_centrality(self):
        """Devuelve un diccionario con el valor eigenvector_centrality para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['eigenvector'] = nx.eigenvector_centrality(g)
            return self.metrics_dict['eigenvector']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Eigenvector: Power iteration failed.")
            return self.empty_dict()

    def pagerank(self, alpha=0.85, epsilon=1e-3):
        """Devuelve un diccionario con el valor pagerank para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['pagerank'] = nx.pagerank(g, alpha=alpha, tol=epsilon)
            return self.metrics_dict['pagerank']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("PageRank: Power iteration failed.")
            return self.empty_dict()

    def betweenness(self, normalized=True):
        """Devuelve un diccionario con el valor betweenness para cada nodo"""
        g = self.g
        self.metrics_dict['betweenness'] = nx.betweenness_centrality(g, normalized=normalized)
        return self.metrics_dict['betweenness']

#STRUCTURAL HOLES:
    def constraint(self):
        """Devuelve un diccionario con el valor constraint para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['constraint'] = constraint(g)
            return self.metrics_dict['constraint']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Constraint:Power iteration failed.")
            return self.empty_dict()

    def effective_size(self):
        """Devuelve un diccionario con el valor effective_size para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['effective_size'] = effective_size(g)
            return self.metrics_dict['effective_size']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Effective size:Power iteration failed.")
            return self.empty_dict() 

#CLUSTERING:
    #V0.0.4
    def average_clustering(self):
        g = self.g
        return nx.average_clustering(g)


    def clustering(self):
        """Devuelve un diccionario con el valor clustering para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['clustering'] = nx.clustering(g)
            return self.metrics_dict['clustering']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Clustering:Power iteration failed.")
            return self.empty_dict()

    def square_clustering(self):
        """Devuelve un diccionario con el valor square_clustering para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['square-clustering'] = nx.square_clustering(g)
            return self.metrics_dict['square-clustering']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Clustering:Power iteration failed.")
            return self.empty_dict()

    def triangles(self):
        """Devuelve un diccionario con el valor triangles para cada nodo"""
        try:
            g = self.g
            self.metrics_dict['triangles'] = nx.triangles(g)
            return self.metrics_dict['triangles']
        except nx.exception.PowerIterationFailedConvergence:
            self.logging_message("Triangles:Power iteration failed.")
            return self.empty_dict()
        except nx.exception.NetworkXNotImplemented:
            self.logging_message("Triangles:No implemented for directed graph in networkx.")
            g = self.get_undirected()
            self.logging_message("Returning results getting from undirected equivalent graph.")
            self.metrics_dict['triangles'] = nx.triangles(g)
            return self.metrics_dict['triangles']

#TRANSITIVITY V0.0.4
    #V0.0.4
    def global_transitivity(self):
        g = self.g.to_undirected()
        return nx.transitivity(g)

    #V0.0.4
    def node_transitivity(self, node):
        """Devuelve la transitividad del subgrafo que contiene en cada vértice al nodo"""
        if self.g.is_directed():
            g = self.g.to_undirected()
        else:
            g = self.g
        node_as_trg = [u for u, v in g.edges() if v == node]
        node_as_src = [v for u, v in g.edges() if u == node]
        nodes_subgraph = [*node_as_trg, *node_as_src, *[node]]
        nodes_subgraph = sorted(list(set(nodes_subgraph)))
        sub_graph = g.subgraph(nodes_subgraph)
        return nx.transitivity(sub_graph)

    #V0.0.4
    def local_transitivity(self):
        """Devuelve un diccionario con el valor local transitivity para cada nodo"""
        g = self.g
        local_transitivity = {}
        for i in g.nodes():
            local_transitivity[i] = self.node_transitivity(i)
        self.metrics_dict['local_transitivity'] = local_transitivity
        return self.metrics_dict['local_transitivity']

#COMUNITIES: (In construction)
    def louvain_communities(self, **kwargs):
        """Devuelve un diccionario con el valor louvain_communities para cada nodo"""
        g = nx.Graph(self.g)
        communities = cm.best_partition(g, **kwargs)
        return communities

    def greedy_modularity_communities(self,**kwargs):
        """Devuelve un diccionario con el valor greedy_communities para cada nodo"""
        g = nx.Graph(self.g)
        c = list(comm.greedy_modularity_communities(g, **kwargs))
        communities = {}
        for i in range(len(c)):
            for j in c[i]:
                communities[j] = i
        return communities

    def k_clique_communities(self, k=2, **kwargs):
        """Devuelve un diccionario con el valor k_clique_communities para cada nodo"""
        if k < 2:
            k = 2
        g = nx.Graph(self.g)
        c = list(comm.k_clique_communities(g, k, **kwargs))
        communities = {}
        for i in range(len(c)):
            for j in c[i]:
                communities[j] = i
        return communities

    def asyn_fluidc(self, k=10, **kwargs):
        """Devuelve un diccionario con el valor asyn_fluidc_communities para cada nodo, k = el más tamaño   
           más pequeño"""
        g = nx.Graph(self.g)
        communities = comm.asyn_fluidc(g, k, **kwargs)
        return self.get_dict(communities)

    def girvan_newman_modularity_communities(self, **kwargs):
        """Devuelve un diccionario con el valor girvan_newman_communities para cada nodo"""
        g = nx.Graph(self.g)
        communities = []
        comp = comm.girvan_newman(g, **kwargs)
        communities = sorted(map(sorted, next(comp)))
        dict_ = {}
        for i in range(len(communities)):
            for j in communities[i]:
                dict_[j] = i
        return dict_

    def kernighan_lin_bisection(self,**kwargs):
        """Devuelve un diccionario con el valor kernighan_communities para cada nodo"""
        g = nx.Graph(self.g)
        communities = comm.kernighan_lin_bisection(g, **kwargs)
        return self.get_dict(communities)

    def asyn_lpa_communities(self, **kwargs):
        """Devuelve un diccionario con el valor asyn_communities para cada nodo"""
        g = nx.Graph(self.g)
        communities = comm.asyn_lpa_communities(g, **kwargs)
        return self.get_dict(communities)

    def label_propagation_communities(self):
        """Devuelve un diccionario con el valor label_propagation_communities para cada nodo"""
        g = nx.Graph(self.g)
        communities = comm.label_propagation_communities(g)
        return self.get_dict(communities)

    def get_dict(self, object_):
        """Crea un diccionario para un objeto, función interna"""
        css = [i for i in object_]
        dict_ = {}
        for i, lg in enumerate(css):
            for node in lg:
                dict_[node] = i
        return dict_

    def get_dc(self, list_):
        """Crea un diccionario desde una lista, función interna"""
        dict_ = {}
        for i in self.g.nodes:
            for k in list_:
                if i in k:
                    dict_[i] = list_.index(k)
        return dict_

    def dict_of_communities(self, algorithm='asyn_fluidc', k=4, **kwargs):
        """Devuelve un diccionario con las comunidades calculadas según el algoritmo específico"""
        """Default = asyn_fluidc, recomendado en networkx"""
        g = self.g
        dict_={}
        if algorithm == 'greedy':
           dict_ = self.greedy_modularity_communities(**kwargs)
        elif algorithm == 'girvan_newman':
           dict_ = self.girvan_newman_modularity_communities(**kwargs)
        #elif algorithm == 'k_clique':
        #   dict_ = self.get_dict(self.k_clique_communities(k=k, **kwargs))
        elif algorithm == 'asyn_fluidc':
           dict_ = self.asyn_fluidc(k=k, **kwargs) 
        elif algorithm == 'kernighan_bisection':
           dict_ = self.kernighan_lin_bisection(**kwargs)
        elif algorithm == 'lpa':
           dict_ = self.label_propagation_communities(**kwargs)
        elif algorithm == 'asyn_lpa':
           dict_ = self.asyn_lpa_communities(**kwargs)
        elif algorithm == 'louvain':
           dict_ = self.louvain_communities(**kwargs)
        self.metrics_dict['communities'] = dict_
        return dict_

    def get_partition(self, dict_of_communities):
        """Devuelve la partición en comunidades de un diccionario de comunidades, uso interno"""
        d = dict_of_communities
        partition = []
        for i in list(set(d.values())):
            sub_part = []
            for j in d.keys():
                if d[j] == i:
                    sub_part.append(j)
            partition.append(sub_part)
        return partition

    def modularity(self, algorithm = 'asyn_fluidc', **kwargs):
        """Devuelve la modularidad según la lista de comunidades"""
        g = self.g
        print("Returning modularity communities with algorithm %s of networkx."%algorithm)
        dict_ = self.dict_of_communities(algorithm=algorithm, **kwargs)
        return comm.modularity(g, self.get_partition(dict_))

    def metrics_df(self, metrics='all', pr_alpha_epsilon=(0.85,1e-3), algorithm='asyn_fluidc', k=10):
        """Devuelve un dataframe de métricas seleccionadas por lista"""
        """Parámetro principal:
           metrics = [nombre de la metrica], all devuelve todas las métricas programadas
        """
        g = self.g
        nodes = [i for i in g.nodes]
        labels = [self.vertex_labels[i] for i in g.nodes]
        #Creación dataframe:
        cols = ['node']
        df = pd.DataFrame(nodes, columns=cols)
        df['labels'] = labels
        
        if metrics == 'all':
            if g.is_directed():
                metrics = ['degree', 'indegree', 'outdegree', 'eccentricity', 'pagerank', 
                           'eigenvector', 'betweenness', 'harmonic', 'closeness',
                           'communities', 'constraint', 'effective_size', 'local_transitivity',
                           'clustering', 'triangles', 'square_clustering']
            else:
                metrics = ['degree', 'eccentricity', 'pagerank', 'eigenvector', 'constraint',
                           'effective_size','harmonic', 'closeness', 'betweenness', 'communities',
                           'clustering', 'triangles', 'local_transitivity', 'square_clustering']

        for i in metrics:
            if i in self.metrics_dict.keys():
                df[i] = self.metrics_dict[i].values()
            else:
                if i == 'degree':
                    df['degree'] = self.degree().values()
                if i == 'indegree':
                    df['indegree'] = self.indegree().values()
                if i == 'outdegree':
                    df['outdegree'] = self.outdegree().values()
                if i == 'eccentricity':
                    df['eccentricity'] = self.eccentricity().values()
                if i == 'pagerank':
                    df['pagerank'] = self.pagerank(alpha=pr_alpha_epsilon[0], epsilon=pr_alpha_epsilon[1]).values()
                if i == 'eigenvector':
                    df['eigenvector'] = self.eigenvector_centrality().values()
                if i == 'harmonic':
                    df['harmonic'] = self.harmonic_centrality().values()
                if i == 'closeness':
                    df['closeness'] = self.closeness_centrality().values()
                if i == 'betweenness':
                    df['betweenness'] = self.betweenness().values()
                if i == 'constraint':
                    df['constraint'] = self.constraint().values()
                if i == 'effective_size':
                    df['effective_size'] = self.effective_size().values()
                if i == 'clustering':
                    df['clustering'] = self.clustering().values()
                if i == 'triangles':
                    df['triangles'] = self.triangles().values()
                if i == 'square_clustering':
                    df['square_clustering'] = self.square_clustering().values()
                if i == 'local_transitivity':
                    df['local_transitivity'] = self.local_transitivity().values()
                if i == 'communities':
                    df['communities_%s'%algorithm] = self.dict_of_communities(algorithm=algorithm, k=k).values()

        return df

    def metrics_csv(self, save_as, metrics='all', index=False, **kwargs):
        """Devuelve las metricas networkx seleccionadas en lista, a un csv"""
        df = self.metrics_df(metrics=metrics, **kwargs)
        df.to_csv(save_as, index=index)

    #V.0.0.4
    def max_min_normalization(self, datalist):
        """Calcula las métricas como normalización max min entre [0 1], min = 0, max = 1"""
        max_value = max(datalist)
        min_value = min(datalist)
        denominator = float(max_value - min_value)
        max_min_list = [float(i - min_value)/ denominator for i in datalist]
        return max_min_list

    #V.0.0.4
    def max_min_metrics_df(self, metrics='all', save_as = 'max_min_metrics.csv', save=True, index=False, **kwargs):
        """Devuelve las metricas seleccionadas networkx en lista con valores max_min, a un csv"""
        df = self.metrics_df(metrics=metrics, **kwargs)
        value_columns = [df.columns.tolist()[i] for i in range(2, len(df.columns.tolist()))]
        for i in value_columns:
            if not(i.startswith('communities')):
                df[i] = self.max_min_normalization(df[i].tolist())
        if save:
            df.to_csv(save_as, index=index)
        return df

    def plot_node_metric(self, metric, tail_number=25, ascending=True, save=False, save_as='figure.png', html=False, html_name='figure.html', figw = 10, figh = 10, color='g'):
        """Crea un gráfico con la metrica seleccionada, muestra por default 25 más altos."""
        plt.close('all')
        df = self.metrics_df(metrics=[metric]).sort_values(metric, ascending=ascending).tail(tail_number)
        ax = df.plot.barh(x='node', y=metric, color=color)
        ax.set_yticklabels(df['labels'])
        ax.set_xlabel('Value')
        ax.set_ylabel('Node')
        ax.figure.set_size_inches(figw, figh)
        if save or html:
            if save:
                plt.savefig("%s_%s"%(metric,save_as))
            if html:
                mpld3.save_html(plt.gcf(),"%s_%s"%(metric,html_name))
        else:
            plt.show()

    def get_type(self, type_name):
        """Devuelve el type, utilizado para el convertidor a .gefx"""
        if type_name == 'int': return "integer"
        elif type_name == 'float': return "double"
        elif type_name == 'bool': return "boolean"
        elif type_name == 'str': return "string"
        
    def return_metric(self, metric='pagerank', **kwargs):
        """Devuelve la metrica solicitada en forma de diccionario, encapsulador"""
        g = self.g
        try:
            if metric in self.metrics_dict.keys():
                metrics = self.metrics_dict[metric]
            else:
                if metric == 'pagerank':
                    metrics = self.pagerank(**kwargs)
                elif metric == 'betweenness':
                    metrics = self.betweenness(**kwargs)
                elif metric == 'eigenvector':
                    metrics = self.eigenvector_centrality()
                elif metric == 'harmonic':
                    metrics = self.harmonic_centrality()
                elif metric == 'closeness':
                    metrics = self.closeness_centrality()
                elif metric == 'eccentricity':
                    metrics = self.eccentricity()
                elif metric == 'degree':
                    metrics = self.degree()
                elif metric == 'constraint':
                    metrics = self.constraint()
                elif metric == 'effective_size':
                    metrics = self.effective_size()
                elif metric == 'clustering':
                    metrics = self.clustering()
                elif metric == 'triangles':
                    metrics = self.triangles()
                elif metric == 'square_clustering':
                    metrics = self.square_clustering()
                elif metric == 'local_transitivity':
                    metrics = self.local_transitivity()
                      
                if nx.is_directed(g): 
                    if metric == 'indegree':
                        metrics = self.indegree()
                    if metric == 'outdegree':
                        metrics = self.outdegree()
                else:
                    if metric in ['outdegree', 'indegree']:
                        print('Warning: Undirected graph hasn\'t outdegree or indegree')
                        metrics = self.degree()
            return metrics
        except UnboundLocalError:
            print("Metric not found, or there is a mistake in name.")
            return self.empty_dict()
                

# --- PLOT METRIC HISTOGRAM:
    #V.0.0.4
    def plot_metric_histogram(self, metric='pagerank', save_as='hist.png', html=False, save=True, html_name ='hist.html', bins=50, **kwargs):
        metrics = self.return_metric(metric=metric, **kwargs)    
        data_x = metrics.values()
        plt.close('all')
        plt.hist(data_x, bins=bins)
        plt.title('Frequency Histogram');
        plt.xlabel(metric)
        plt.ylabel('number of nodes')
        if save or html:
            if save:
                plt.savefig(save_as)
            if html:
                mpld3.save_html(plt.gcf(), html_name)
        else:
            plt.show()

#======================== FILE EXTENSION ====================================================

    def return_extension(self, file_name):
        f_name = ""
        f_ext = ""
        i = file_name.rfind(".")
        if not(i==0):
            n = len(file_name)
            j = n - i - 1
            f_name = file_name[0:i]
            f_ext = file_name[-j:]
        return f_ext

    #V.0.0.4
    def output_name(self, save_as, extension):
        output_name = ''
        if self.return_extension(save_as) == extension:
            output_name = save_as
        else:
            output_name = save_as + '.' + extension
        return output_name

#========================GML =======================================================
    #V.0.0.4
    def write_gml(self, save_as='graph.gml', **kwargs):
        g = self.g
        output_name=self.output_name(save_as, 'gml')
        nx.write_gml(g, output_name)

#========================GEPHI =======================================================

    def write_gexf(self, save_as='g_graph.gexf'):
        """Escribe un archivo gexf utilizando la librería interna de networkx (requiere de atributos)"""
        g = self.g
        output_name=self.output_name(save_as, 'gexf')
        nx.write_gexf(g, output_name)

    def save_to_gephi(self, save_as='graph.gexf', description='', mode='static'):
        """Escribe un archivo gexf, utilizando los parámetros definidos en este wrapper"""
        g = self.g
        nodes = list(g.nodes)
        edges_list = [i.split(',') for i in nx.generate_edgelist(g, delimiter=',', data=['weight'])]
        labels = self.vertex_labels
        directed = nx.is_directed(g)
        graph_mode = mode

        #graph type:
        graph_type = ''
        if directed:
            graph_type = 'directed'
        else:
            graph_type = 'undirected'
    
        #meta info
        gexf = """<?xml version="1.0" encoding="UTF-8"?>\n<gexf version="1.2" xmlns="http://www.gexf.net/1.2draft" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.gexf.net/1.2draft http://www.gexf.net/1.2draft/gexf.xsd">\n"""
        meta = "  <meta lastmodifieddate=\"%s\">\n"%date.today().strftime("%Y-%m-%d")
        creator = "    <creator>%s</creator>\n"%('data_graph')
        descript = "    <description>%s</description>\n"%(description) 
        close_meta = "  </meta>\n"  
        gexf += meta + creator + descript + close_meta
        gexf_close = "</gexf>"

        #graph info:
        graph = "  <graph mode=\"%s\" defaultedgetype=\"%s\">\n"%(graph_mode, graph_type)
        graph_close = "  </graph>\n"

        #if there are nodes attributes:
        node_attr = "   <attributes class=\"node\" mode=\"%s\">\n"%mode
        node_attr_close = "   </attributes>\n"
        node_attributes = self.list_nodes_attributes()
        dic_node_attr = self.get_dict_node_attrib()
        attr_head = ""
        if len(node_attributes)>0:
            attr_head = node_attr
            for i in range(len(node_attributes)):
                type_value = self.get_type(type(dic_node_attr[node_attributes[i]][0]).__name__)
                attr_head += "      <attribute id=\"%d\" title=\"%s\" type=\"%s\" />\n"%(i, node_attributes[i], type_value)
            attr_head += node_attr_close
                
        #Get nodes and labels
        node_str = "   <nodes>\n"
        for i in nodes:
            if len(node_attributes)>0:
                node_str += "     <node id=\"%d\" label=\"%s\">\n"%(i,labels[i])
                node_str += "         <attvalues>\n"
                for m in range(len(node_attributes)):
                    value = dic_node_attr[node_attributes[m]][i]
                    node_str +="            <attvalue for=\"%d\" value=\"%s\" /> \n"%(m,value)
                node_str += "         </attvalues>\n"  
                node_str += "     </node>\n"
            else:
                node_str += "     <node id=\"%d\" label=\"%s\" />\n"%(i,labels[i])
        node_str += "   </nodes>\n"

        #Get edges:
        edge_str = "   <edges>\n"
        k = 0
        for i in edges_list:
            edge_str += "     <edge id=\"%d\" source=\"%s\" target=\"%s\" weight=\"%s\" />\n"%(k, i[0], i[1], i[2])
            k += 1
        edge_str += "    </edges>\n"
    
        #Output string:
        gexf += graph + attr_head + node_str + edge_str + graph_close + gexf_close

        if self.return_extension(save_as) == 'gexf':
            output_name = save_as
        else:
            output_name = save_as + '.gexf'
    
        #save file:
        with open(output_name, 'w', encoding='utf-8') as file:
            file.write(gexf)

# =================================DRAW GRAPH ===============================================

    # ------ SET LAYOUT:
    def default_layout(self):
        """Devuelve un layout para dibujar el grafo con networkx, por default es random_layout"""
        g = self.g
        return nx.random_layout(g)

    def set_nx_layout(self, layout='random', **kwargs):
        """Configura el layout para dibujar el grafo con networkx"""
        g = self.g
        if layout == 'random':
            self.pos = nx.random_layout(g)
        elif layout == 'shell':
            self.pos = nx.shell_layout(g, **kwargs)
        elif layout == 'spring':
            self.pos = nx.spring_layout(g, **kwargs)
        elif layout == 'circular':
            self.pos = nx.circular_layout(g, **kwargs)
        elif layout == 'kamada_kawai':
            self.pos = nx.kamada_kawai_layout(g, **kwargs)
        elif layout == 'bipartite':
            self.pos = nx.bipartite_layout(g, **kwargs)
        elif layout == 'spectral':
            self.pos = nx.spectral_layout(g, **kwargs)
        elif layout == 'spiral':
            self.pos = nx.spiral_layout(g, **kwargs)
        elif layout == 'planar':
            self.pos = nx.planar_layout(g, **kwargs)

    def rescale_nx_pos(self):
        """Devuelve una nueva escala para las coordenadas de los nodos en el dibujo del grafo"""
        self.pos = nx.rescale_layout(g, pos = self.pos)

    def get_graph_pos(self):
        """Devuelve las coordenadas en el dibujo de cada uno de los nodos del grafo"""
        return self.pos

    def set_graph_pos(self, layout):
        """Configura un nuevo layout de coordendadas de grafos"""
        self.pos = layout

    #-----DRAW GRAPH:
    def draw_graph(self, g = None, html=False, save=False, html_name ='graph1.html', save_as='graph1.png', **kwargs):
        """Dibuja el grafo utilizando las propiedades de networkx, puede salvar a .png o a .html"""
        plt.close('all')
        if g == None:
            g = self.g
        else:
            g = g
        if self.pos == None:
            pos = self.set_nx_layout(layout='spring')
        else:
            pos = self.pos
        nx.draw(g, pos, **kwargs)
        plt.draw()
        plt.axis('off')
        if save or html:
            if save:
                plt.savefig(save_as)
            if html:
                mpld3.save_html(plt.gcf(), html_name)
        else:
            plt.show()

    #-----DRAW GRAPH WITH METRICS:
    def draw_graph_metrics(self,
                        html=False, 
                        save=True, 
                        html_name ='graph1.html', 
                        save_as='graph1.png', 
                        metric='pagerank', 
                        size_factor = 10000,
                        with_labels = False,
                        with_node_label = True,
                        with_values = False,
                        first_n_values = 5,
                        node_size = 50,
                        normal_node_size = True,
                        cmap = 'YlOrRd',
                        **kwargs):

        """Dibuja el grafo seleccionado una métrica para colorear los nodos"""

        g = self.g
        
        metrics = self.return_metric(metric=metric, **kwargs)

        labels = self.vertex_labels
        node_label = dict()

        k = first_n_values
        ordered_list_values = sorted([value for value in metrics.values()], reverse=True)[0:k]
    
        for key in metrics.keys():
            value = metrics.get(key)
            if value in ordered_list_values:
               if with_node_label and with_values:
                  node_label[key] = "(%s, %0.2f)"%(labels[key],value)
               elif with_values and not with_node_label:
                   node_label[key] = "(%0.2f)"%(value)
               elif with_node_label and not with_values:
                   node_label[key] = "(%s)"%(labels[key])
               else:
                   node_label[key] = ''

        allowed = ['pagerank', 'betweenness', 'eigenvector', 'eccentricity', 'degree', 'indegree', 'outdegree', 'harmonic', 'closeness', 'triangles', 'constraint', 'effective_size', 'clustering',
        'square_clustering', 'local_transitivity']
                
        if metric in allowed:
            if normal_node_size:
                self.draw_graph(g=g, html=html, save=save, html_name=html_name, save_as=save_as, nodelist=[i for i in metrics.keys()], node_size = node_size,labels=node_label, with_labels=with_labels, node_color=[i for i in metrics.values()], cmap=plt.get_cmap(cmap), **kwargs)
            else:
                self.draw_graph(g=g, html=html, save=save, html_name=html_name, save_as=save_as, nodelist=[i for i in metrics.keys()], node_size=[v * size_factor for v in metrics.values()], labels=node_label, with_labels=with_labels, node_color=[i for i in metrics.values()], cmap=plt.get_cmap(cmap), **kwargs)


    #-----DRAW GRAPH WITH COMMUNITIES:
    def draw_graph_community(self,
                        html=False, 
                        save=True, 
                        html_name ='graph_communities.html', 
                        save_as='graph_communities.png', 
                        community='asyn_fluidc', 
                        with_labels = False,
                        with_node_label = True,
                        with_values = False,
                        node_size = 50,
                        cmap = 'YlOrRd',
                        **kwargs):

        """Dibuja el grafo seleccionado un algoritmo para colorear los nodos"""

        g = self.g
        labels = self.vertex_labels
      
        node_label = {}
        if community in self.metrics_dict.keys():
            commu_ = self.metrics_dict[community]
        else:
            commu_ = self.dict_of_communities(algorithm=community, **kwargs)

        for key in commu_.keys():
            node_label[key] = "%s"%labels[key]

        print("Number of communities %s detected: %d"%(community, len(list(set(commu_.values())))))

        self.draw_graph(g=g, html=html, save=save, html_name=html_name, save_as=save_as, node_size = node_size,labels=node_label, with_labels=with_labels, node_color=[i for i in commu_.values()], cmap=plt.get_cmap(cmap), **kwargs)

if __name__ == '__main__':
    print("Module create_graph")
    
    
