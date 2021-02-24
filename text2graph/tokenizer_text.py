#!/usr/bin/python3

# =============================================================================
# Tokenizer of Text Tools and Utils
# =============================================================================
#
# Miscellaneous utility functions to be used with text to get tokens and
# source and target nodes for concepts networks
# @Author: Brayan Rodriguez <bradrd2009jp@gmail.com>
# @Organization: LIIT-UNED 2020

#TODO:
#Create a best tokenizer model and lematization for spanish or improve
#use of spycy and nltk.
#Maybe use a automaticed algorithm of learning to stemming and lematization.

#Convertidor de texto a tokens
#Import:
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, treebank
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer

from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer

import nltk
import string
import re
from string import digits
import locale
import pandas as pd

#Experimental:
import spacy

__all__ = ['tokenize_text']

#Constantes:

PREPOSITIONS = ['a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de'
                'desde', 'en', 'entre', 'para', 'por', 'segun', 'sin',
                'so', 'sobre', 'tras']
NO_SEMANTIC_WORDS = ['mas', 'asi', 'menos', 'ser', 'estar', 'ello', 'mientras', 'despues', 
                'tanto', 'mismo', 'parecer', 'tambien', 'si', 'no', 'etcetera', 'hacia', 
                'durante', 'decir', 'desear', 'recitar', 'cerca', 'lejos', 'entonces', 
                'luego', 'hola', 'ningun', 'primer', 'primero', 'atras', 'delante', 'ademas']
ABBREVIATIONS = ['etc', 'sr', 'sres', 'sras', 'srta']
ENCLITIC_PRONOUNS = ['me', 'se', 'te', 'nos', 'le', 'la', 'lo', 'los', 'las']
PUNCTUATION_SIGN = [i for i in string.punctuation]
CURRENCIES_SYMB = ['$', '€', '¢', '¥']
OTHERS_SYMB = ['...', "\"", "`", "''", "``", "¿", "?", "º", "¡", "“", "*", "-","_", "”" ]
NOUNS_ES_FINISHED_IN_S = ['pais', 'virus', 'dios', 'coronavirus', 'viernes']
NOUNS_ES_FINISHED_IN_R = ['mar', 'par']
NOUNS_ES_FINISHED_IN_RIA = ['historia', 'histeria', 'alegria']
NOUNS_ES_FINISHED_IN_TO = ['manifiesto', 'movimiento']
NOUNS_ES_FINISHED_IN_RO = ['carnero', 'astillero']
NOUNS_OF_MONTH = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'setiembre', 'octubre', 'noviembre', 'diciembre']
NAMES=['figueres', 'chavarria', 'chaves', 'cespedes', 'maria', 'jose', 'carlos', 'luis', 'echeverria', 'arias']

class tokenize_text():
    def __init__(self, text, language ='spanish', with_stopwords=False):
        self.language = language
        self.stemmer = self.stemmer()
        self.text = text.rstrip('\n') #Elimina saltos de carro
        self.text = self.remove_emoji(self.text) #Elimina emojis
        self.text = self.text.replace(u'\u200d️', '') #Elimina simbolos
        self.text = self.text.translate({ord(k): None for k in digits}) #Elimina números
        if language == 'spanish':
            self.text = self.normalize_spanish_text()

        if with_stopwords:
            self.token = self.tokenize()
        else:
            self.token = self.tokenize_without_stopwords()

    def print(self):
        print(self.text)

    def remove_emoji(self, text):
        try:
            emoji_pattern = re.compile(u'['
            u'\U0001F300-\U0001F64F'
            u'\U0001F680-\U0001F6FF'
            u'\u2600-\u26FF\u2700-\u27BF]+', 
            re.UNICODE)   
            return emoji_pattern.sub('', text) #no emoji text
        except re.error:
            # Narrow UCS-2 build
            emoji_pattern = re.compile(
            u"(\ud83d[\ude00-\ude4f])|"  # emoticons
            u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
            u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
            u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
            u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
            "+", flags=re.UNICODE)
            return emoji_pattern.sub('', text)
        
    def tweet_tokenize(self):
        tokenizer = TweetTokenizer()
        text = self.lower()
        return tokenizer.tokenize(text)
    
    def tokenize(self):
        punct_sign = CURRENCIES_SYMB + OTHERS_SYMB + PUNCTUATION_SIGN
        stop_words = set(stopwords.words(self.language))
        text = word_tokenize(self.normalize(self.lower()))
        new_text = list()
        for w in text:
            #if (w not in stop_words) and (w not in punct_sign):
            if (w not in punct_sign) and (w not in NO_SEMANTIC_WORDS) and (w not in ABBREVIATIONS):
                new_text.append(w)
        return new_text
    
    def tokenize_without_stopwords(self):
        punct_sign = CURRENCIES_SYMB + OTHERS_SYMB + PUNCTUATION_SIGN
        stop_words = set (stopwords.words(self.language))
        text = word_tokenize(self.normalize(self.lower()))
        new_text = list()
        for w in text:
            if (w not in stop_words) and (w not in punct_sign) and (w not in PREPOSITIONS) and (w not in NO_SEMANTIC_WORDS) and (w not in ABBREVIATIONS):
                new_text.append(w)
        return new_text
    
    def lower(self):
        return self.text.lower()
    
    #Limpieza:
    def normalize(self, s):
        replacements = (
            ("á", "a"),
            ("é", "e"),
            ("í", "i"),
            ("ó", "o"),
            ("ú", "u"),
            ("ñ", "_n"),
            ("Ã±", "_n"),
            ("i±", "_n"),
            ("iÂ±", "i"),
            ("Ã³", "o"),
            ("Ã­ ", "i "),
            ("Ã¡", "a"),
            ("Ã©", "e"),
            ("Ãº", "u"),
            ("." , ""),
            ("!" , ""),
            ("¡" , ""),
            ("?" , ""),
            ("_*", ""),
            ("¿", ""),
            ("-*", ""),
            ("*", ""),
            ("--", ""),
            ("costa rica", "costarica")
        )
        for a, b in replacements:
            s = s.replace(a, b)
        return s
    
    def normalize_spanish_text(self):
        text = word_tokenize(self.lower())
        lex = []
        for word in text:
            lex.append(word.lower())
        return self.normalize(' '.join(lex))

    def get_token_frequency_df(self, with_stopwords=False):
        df = pd.DataFrame()
        if with_stopwords:
            token_list = self.tokenize()
        else:
            token_list = self.tokenize_without_stopwords()
        if self.language=='spanish':
            words, pos, lemma = self.filter_spanish_token_list_with_lemma(token_list)   
        else:
            words, pos, lemma = self.lematizing_text(token_list)
        stem = self.stemming_token(words)
        count = [stem.count(x) for x in stem]
        df['word']= words
        df['stem'] = stem
        df['pos'] = pos
        df['labels'] = lemma
        df['stem_count'] = count
        return df

    def get_source_target_graph(self, gap=2):
        source = list()
        target = list()
        df = self.get_token_frequency_df()
        token = df['labels'].tolist() 
        for i in range(len(token) - 1 - gap):
            source.append(token[i])
            target.append(token[i + 1 + gap])
        return source, target
            

    #TODO: Falta decidir si se va a trabajar con a documento de constantes b Radicacion o reemplazo de radicación con palabra más frecuente c verbos irregulares devolver el token.
    
    def stemmer(self, stemmer_type='Snowball', ignore_stopwords='True'):
        if stemmer_type == 'Porter':
            stemmer = PorterStemmer()
        elif stemmer_type == 'Snowball':
            stemmer = SnowballStemmer(language=self.language, ignore_stopwords=ignore_stopwords)
        return stemmer

    def stemming_token(self, token_list):
        token_stm = [self.stemmer.stem(i) for i in token_list]
        return token_stm

    def stemming_word(self, word):
        return self.stemmer.stem(word)
    
    def lematizing_text(self, token_list):
        word = list()
        lemma = list()
        pos = list()
        wnl = WordNetLemmatizer()
        for w in token_list:
             word.append(w)
             lemma.append(wnl.lemmatize(w))
             pos.append(nltk.pos_tag(w)[0][1])
        return word, pos, lemma

    def filter_spanish_token_list_with_lemma(self, token_list, model='es_core_news_md', p_e = ENCLITIC_PRONOUNS):
        nlp = spacy.load(model)
        tk_list = list()
        pos_list = list()
        lm_list = list()
        for i in token_list:
            i = self.normalize(i)
            doc = nlp(i)
            pos = doc[0].pos_
            word_lemma = doc[0].lemma_                
            if i[-3:] == 'ria' and i not in NOUNS_ES_FINISHED_IN_RIA and i not in NAMES:
                   word_lemma = i[:-2]
                   pos = 'VERB'
            if i[-2:] == 'ro' and  nlp(self.stemming_word(i)+'ar')[0].pos_ == 'VERB' and i not in NOUNS_OF_MONTH and i not in NOUNS_ES_FINISHED_IN_RO:
                 word_lemma =  self.stemming_word(i)+'ar'
                 pos = 'VERB'
            if i[-2:] == 'to' and  nlp(self.stemming_word(i)+'ar')[0].pos_ == 'VERB' and self.stemming_word(i)+'ar' not in NOUNS_ES_FINISHED_IN_R and i not in NOUNS_ES_FINISHED_IN_TO:
                 word_lemma =  self.stemming_word(i)+'ar'
                 pos = 'VERB'
            if i[-3:] == 'rio' and  nlp(i[:-3]+'ir')[0].pos_ == 'VERB':
                 word_lemma =  i[:-2]+'ir'
                 pos = 'VERB'
            if pos == 'NOUN' or pos == 'PROPN' and word_lemma not in NO_SEMANTIC_WORDS and i not in NAMES:
                if word_lemma[-1:] is 's' and word_lemma not in NOUNS_ES_FINISHED_IN_S:
                   word_lemma = word_lemma[:-1]
                if word_lemma[-2:] is ('lo' or 'no') and nlp(word_lemma[:-2])[0].pos_=='VERB':
                   word_lemma = word_lemma[:-2]
                   pos = 'VERB'
                if word_lemma[-1:] is 'r' and (i[-1:]=='o' or i[-1:]=='a' or i[-2:]=='as') and word_lemma not in NOUNS_ES_FINISHED_IN_R:
                    word_lemma = i
                if word_lemma[-2:] is 'nt':
                    word_lemma += 'e'
                if word_lemma[-2:] is 'j' and (i[-2:]=='je' or i[-3:]=='jes') :
                    word_lemma += 'e'
            if pos == 'VERB' and i not in NAMES:
                if word_lemma[-2:] in p_e:
                    word_lemma = word_lemma[:-2]
                elif (word_lemma[-3:] in p_e):
                    word_lemma = word_lemma[:-3]
                else: word_lemma = word_lemma
            if word_lemma not in NO_SEMANTIC_WORDS and pos not in ['AUX','DET','INTJ','ADP', 'ADV', 'SCONJ', 'CCONJ', 'NUM', 'PUNCT']:
                tk_list.append(i)
                pos_list.append(pos)
                lm_list.append(self.normalize(word_lemma))
        return tk_list, pos_list, lm_list

if __name__=='__main__':
    print("Tokenizer")


