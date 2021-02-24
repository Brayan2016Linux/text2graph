#!/usr/bin/python3

# =============================================================================
# Html Utils
# =============================================================================
#
# Miscellaneous utility functions to be used with html links.
# @Author: Brayan Rodriguez <bradrd2009jp@gmail.com>
# @Organization: LIIT-UNED 2020



#Convertidor de texto a tokens
#Import:
import requests
from bs4 import BeautifulSoup

import tldextract
from urllib.parse import urlparse, parse_qs

import pandas as pd

__all__ = ['get_html_text', 'search']

#add elements to eliminate:
blacklist = ['[document]', 'nav', 'ul', 'li', 'nonscript', 'header', 'meta', 'head', 'input', 'script', 'style', 'span', 'a', 'strong', 'path', 'title', 'img', 'iframe', 'svg', 'use', 'class', 'src', 'br']

def get_html_text(url, parser='html.parser'):
    res = requests.get(url)
    html_page = res.content
    soup = BeautifulSoup(html_page, parser)
    title = soup.title.string
    for script in soup(blacklist):
        script.decompose()
    output = title

    for p_tag in soup.find_all('p'):
        output += '\n' + p_tag.text
    #for li_tag in soup.find_all('li'):
    #    output += '\n' + li_tag.text
    #for div_tag in soup.find_all('div'):
    #    output += '\n' + div_tag.text

    return output

def filter_result(link, search_engine):
    lk = ""
    try:
        if link.startswith('/url?'):
            o = urlparse(link, 'http')
            link = parse_qs(o.query)['q'][0]
        o = urlparse(link, 'http')
        if search_engine == 'google':
            if o.netloc and search_engine not in o.netloc:
                lk = link
            else:
                lk = 'None'
        if search_engine == 'yahoo':
            domain = tldextract.extract(link).domain
            if domain not in ['yahoo', 'uservoice', 'verizonmedia']:
                lk = link
            else:
                lk = 'None'
        return lk
    except Exception:
        pass


def search(query, search_in='google', timeout=None, **kwargs):
    domain_ext = 'com'
    html_parser = 'html.parser'

    if search_in == 'google':
        search_engine = 'www.google'
        query_format = 'search?q='
    elif search_in == 'yahoo':
        search_engine = 'search.yahoo'
        query_format = 'search?p='

    for key, value in kwargs.items():
        if seach_in == 'google' and key == 'country' and value == 'CR' or 'cr':
            domain_ext = 'co.cr' 
        if key == 'parser':
            html_parser = value
    
    req = requests.get('https://%s.%s/%s%s'%(search_engine, domain_ext, query_format, query), timeout=timeout)

    soup = BeautifulSoup(req.content, html_parser)
    hrefs = soup.find_all('a', href=True)

    url_title = list()
    url_link = list()

    for a in hrefs:
       link = filter_result(a['href'], search_in)
       if link is not 'None' and link.startswith('http'):
            if a.text.find('Cached'):
                url_title.append(a.text)
                url_link.append(link)

    df = pd.DataFrame()
    df['title'] = url_title
    df['link'] = url_link

    return df
    

if __name__=='__main__':
    print("html_text")


