#!/usr/bin/python3


import text2graph.data_graph as tg
import text2graph.tokenizer_text as tkt
import text2graph.ct_extractor as ce
import text2graph.html_text as html


if __name__ == '__main__':

   
    #=========== Testeo de captura con api ========
    token = 'UH27nsU12uhrk5ADA1JlxaEWKhBL4Z04DtPur5ZI'
    ct_dt = ce.get_ct_data(token)

    #Ejemplos:
    address1='http://www.repretel.com/actualidad/investigan-afectacion-de-la-salud-mental-en-tiempos-de-covid-19--197242'
    address2='https://www.diarioextra.com/Noticia/detalle/428549/-160-millones-en-salarios-a-jueces-de-competencia?fbclid=IwAR2GeuPIcZFB2yTg6dF_uj_WNsgSzZzLTmOYuT6DQ9MIsvU5-7XuxmW4ua8'
    address3='https://www.larepublica.net/noticia/ucr-conversa-con-empresas-aliadas-para-determinar-cantidad-de-ventiladores-que-donaria-a-la-caja'
    address4="https://www.facebook.com/167533350116993/posts/856861164517538"
    address5="https://www.teletica.com/sucesos/antisociales-rocian-gasolina-a-local-comercial-y-le-prenden-fuego_270110"
    address6="https://www.ameliarueda.com/nota/figueres-alvarado-democracia-amenaza-gobiernos-camina-noticias-costarica"
    address7="https://www.nacion.com/el-pais/salud/muertes-por-coronavirus-en-costa-rica/XW6K4QJISRFTZF5Q7HQ6NFN4HQ/story/"

    addr = address6
    ct_link = ct_dt.links(link=addr) #llamar una única vez, sino aparece error 429
    text1 = ct_link.title() + ct_link.message()
    dom = ct_link.domain()
    print("Dominio: %s"%dom)
    print("Plataforma: %s"%ct_link.platform_id())
    print("Facebook URL: %s"%ct_link.post_url())
    tipo_media = ct_link.media_type()
    print("Tipo media: %s"%tipo_media)
    print(ct_link.statistics_df().T)
    #ct_link.statistics_df().to_csv('news_ctdata.csv')
    #print(get_html_text(addr))

    """
    p_id2 = "100237323349361_3834864693219920"
    p_id1 = "573016139398622_3088000491233495"
    p_id3 = "234653643236303_3686440328057600"

    pid = p_id1
    ct_post = ct_dt.post(pid, includeHistory = False)
    title = ct_post.title()
    message = ct_post.message()
    print(ct_post.post_url())

    #print(message)
    text3 = title + ' ' + message
    """
    #=========================fin====================

    name_file="text1_"
    text = text1 + html.get_html_text(addr)
    gap = 3

    #Create Tokens from text:
    token_text = tkt.tokenize_text(text, with_stopwords=False)
    source, target = token_text.get_source_target_graph(gap=gap)
    
    #Create Graph:
    text_graph = tg(source, target)
    g = text_graph.g
    print(text_graph.get_node_frequency_df())
    text_graph.plot_node_frequency(html=True)
    #text_graph.plot_node_metric(metric='betweenness')
    text_graph.plot_node_metric(metric='pagerank', html=False)
    text_graph.save_to_gephi()
    text_graph.save_edge_list()
    #print(text_graph.get_frequency())
    
    #print(text_graph.network_modularity())
    #dfm = text_graph.metrics_df(metrics=['pagerank', 'degree', 'betweenness'])
    #dfm = text_graph.metrics_csv('metrics.csv')


    #adjacency_matrix:
    #df2 = text_graph.adj_matrix_dataframe()
    #df2.to_csv('adjacency.csv', index=False)
    
    #INFO
    print("Nodos: %d\nAristas: %d"%(text_graph.number_of_vertex(), text_graph.number_of_edges()))
    
    #Set Layouts:
    #text_graph.set_nx_layout(layout='spring', k=0.15, iterations=50)
    #text_graph.set_nx_layout(layout='random')
    #text_graph.set_nx_layout(layout='circular')
    #text_graph.set_nx_layout(layout='spectral')
    #text_graph.set_nx_layout(layout='spiral')
    #text_graph.set_nx_layout(layout='shell')
    #text_graph.set_nx_layout(layout='bipartite', nodes = g.nodes())
    text_graph.set_nx_layout(layout='spring')
    #text_graph.set_nx_layout(layout='kamada_kawai', scale=2)
    
    #Draw Direct
    text_graph.draw_graph_metrics(html=False, metric='pagerank', with_labels=True, with_values=False)
    

    

