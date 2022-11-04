import streamlit as st
from pymed import PubMed
import pandas as pd
import sklearn
import pickle
import os
from datetime import datetime
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from itertools import tee, islice, chain
from urllib.error import HTTPError
#header
st.title('GolemPharm')
st.write('')

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a number of abstracts ',
    0, 4999, 4000, 1)

#input box
int_put = st.text_input('Type disease here:')


if int_put:
 try:
  with st.spinner('Please wait...'):
    text = int_put
    max1 =int(add_slider)
  #wporwadzenie zapytania do pubmed
    pubmed = PubMed(tool="golem", email="p.karabowicz@gmail.com")
    my_api_key = '3f16e1d2a0aae15e5d9dfc6610fbd0a81709'
    pubmed.parameters.update({'api_key': my_api_key})
    pubmed._rateLimit = 10
    results1 = pubmed.query(text, max_results=max1)

    #przeksztalcenie wynikow zapytania na data frame
    lista_abstract_3=[]
    for i in results1:
        lista_abstract_3.append(i.abstract)

    df_abstract = pd.DataFrame(lista_abstract_3, columns = ['abstracts'])
    df_abstract['abstracts_lower'] = df_abstract['abstracts'].str.lower()
    df_abstract_1 = df_abstract.dropna() #datafraame wynikow do analizy

    rnd = pickle.load(open('./finalized_model2022.sav', 'rb'))

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
    result1=count_vect.fit_transform(df_abstract_1['abstracts_lower'])
    result2 = rnd.predict(result1)
    df_abstract_1['class'] = result2 # dataframe z wynikami
    len_df = len(result2)

    from gensim.models import Word2Vec
    from nltk.corpus import stopwords
    #przygotowanie tekstu do osadzania slow
    stop = stopwords.words('english')

    from nltk.tokenize import word_tokenize
    df_abstract_1['abstracts_stop'] = df_abstract_1['abstracts_lower'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    df_abstract_1['tokenized'] = df_abstract_1.apply(lambda row: nltk.word_tokenize(row['abstracts_lower']), axis=1)
    model_ted1 = Word2Vec(sentences=df_abstract_1['tokenized'], vector_size=200, window=10, min_count=1, workers=4, sg=0)

    #ekstrackja slow najbardziej podobnych do protein i target
    keys = ['protein', 'target']
    most_sim = model_ted1.wv.most_similar(positive = keys, topn=1000)
    #utworzenie tablicy ze slownika
    most_sim_key = []
    for w, n in most_sim:
        most_sim_key.append(w)
    #tagowanie tekstu i filtrowanie wedlug tagow
    post_tag_list = nltk.pos_tag(most_sim_key)

    listNN = []
    for w , k in post_tag_list:
        if k == "NN":
            listNN.append(w)

    listJJ = []
    for w , k in post_tag_list:
        if k == "JJ":
            listJJ.append(w)

    lista = [listNN, listJJ]

    flat_list = []
    for sublist in lista:
        for item in sublist:
            flat_list.append(item)

    db_bialka = pd.read_csv('./bialka.csv')
    wt2 = db_bialka['Gene names'].str.lower().tolist()

    tablica_in =[] #lista wyekstrahowanych bialek
    for w in flat_list:
        if w in wt2:
            tablica_in.append(w)

    tablica_len = []
    tablica_wyn = []
    tablica_num = []
    for w in tablica_in:
        g = df_abstract_1[df_abstract_1['abstracts_lower'].str.contains(w)]
        p = (g['class'].sum()/len(g['class']))*100
        tablica_wyn.append(p)
        tablica_len.append(len(g['class']))
        tablica_num.append(g['class'].sum())

    tablica_wynr = [ round(elem, 2) for elem in tablica_wyn ]


    dict = {'Gen': tablica_in, 'percent positive': tablica_wynr,'number of positive abstract':
        tablica_num, 'number of abstracts': tablica_len}

    df_index = pd.DataFrame(dict)
    df_index = df_index.sort_values('number of positive abstract', ascending = False)
    df_index = df_index.reset_index(drop=True)
    df_index['Gen'] = df_index['Gen'].str.upper()
    df_index = df_index.shift()[1:]
    df_index = df_index.round(2)
    #df_index
    st.dataframe(df_index.style.format({'percent positive':'{:.2f}',
    'number of positive abstract': "{:.0f}", 'number of abstracts': "{:.0f}"}))

    def convert_df(df):
      return df.to_csv(index=False).encode('utf-8')
    csv = convert_df(df_index)
    st.download_button(
    "Press to Download",
    csv,
    "file.csv",
    "text/csv",
    key='download-csv')
 except HTTPError:
  st.write('PubMed server overload! Try again later, please')
