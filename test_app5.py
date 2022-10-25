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
import pandas as pd
from itertools import tee, islice, chain

#nagłowek
st.title('GoLem Pharm')
st.write('')

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a number of abstracts ',
    0, 10000, 1000, 1)

#input box
int_put = st.text_input('Ask about your disease here:')
if int_put:
    st.write('your results for request: ', int_put)
    text = int_put
    max1 =int(add_slider)  # ilosc zapytan ze slidera

    #wporwadzenie zapytania do pubmed
    pubmed = PubMed(tool="MyTool", email="p.karabowicz@gmail.com")
    results1 = pubmed.query(text, max_results=max1)

    #przeksztalcenie wynikow zapytania na data frame
    lista_abstract_3=[]
    for i in results1:
        lista_abstract_3.append(i.abstract)
    import pandas as pd
    df_abstract = pd.DataFrame(lista_abstract_3, columns = ['abstracts'])
    df_abstract['abstracts_lower'] = df_abstract['abstracts'].str.lower()
    df_abstract_1 = df_abstract.dropna() #datafraame wynikow do analizy


    import pickle
    rnd = pickle.load(open('./finalized_model2022.sav', 'rb'))

    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
    result1=count_vect.fit_transform(df_abstract_1['abstracts_lower'])
    result2 = rnd.predict(result1)
    df_abstract_1['class'] = result2 # dataframe z wynikami
    len_df = len(result2)

#unsupervised learning
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

    # predykcja dla zapytan z tablica_in
    import sklearn
    from sklearn.feature_extraction.text import CountVectorizer
    import time

    def query(list_target):
        pubmed = PubMed(tool="MyTool", email="p.karabowicz@gmail.com")
        lista=[]
        for w in list_target:
            time.sleep(25)
            lista.append(pubmed.query(w, max_results=20))
        return lista

    def lista_bastract_pred1(lista):
        lista_abstract_pred=[]
        for n in lista:
                lista_abstract_pred.append(n.abstract)
        return lista_abstract_pred

    def percent_true(ynew2):
        percent_true = round((list(ynew2).count(1))/len(list(ynew2))*100,1)
        return percent_true

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    import pickle
    #rnd = pickle.load(open('C:\\Users\\UMB\\Desktop\\drugforest\\drugforest\\static\\finalized_model_32.sav', 'rb'))

    a = query(tablica_in)
    d = []
    for w in a:
        b = lista_bastract_pred1(w) ## ziterować
        time.sleep(1)
        d.append(b)

    d2=[]
    for g in d:
        #d1 = list(filter(None.__ne__, g))
        d1 = list(filter(None, g))
        d2.append(d1)

    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=100)
    lista_y=[]
    for w in d2:
        result1=count_vect.fit_transform(w)
        result2 = rnd.predict(result1)
        lista_y.append(result2)

    import numpy as np
    yy = []
    for y in lista_y:
        yy.append(percent_true(y))

    yy = list(np.around(np.array(yy),0))

    import pandas as pd
    data_tuples = list(zip(tablica_in,yy))
    df_list = pd.DataFrame(data_tuples, columns=['Protein','Score'])

    df_list = df_list.sort_values('Score', ascending = False)
    #df_list = df_list.reset_index()

    df_abstract_1 = df_abstract_1[['abstracts', 'class']]
    #st.write(df_list[['Protein', 'Score']])
    df_index = df_list.reset_index(drop=True)
    st.write(df_index.shift()[1:])

    #drug
    df_drug = pd.read_csv('./Products1.csv')
    list_drug = set(list(df_drug['ActiveIngredient']))
    list_drug = list(list_drug)
    list_drug_lower = [x.lower() for x in list_drug]


    zapyt = df_index['Protein'][0:10]
    wynik = []
    for w in zapyt:
    #st.write('your results for request: ', zapyt)
        text = w
        #max1 =int(add_slider)	# ilosc zapytan ze slidera
        max1 = 200
        drugs_count = 10
        drugs_count1 = int(drugs_count)
        time.sleep(25)
        pubmed = PubMed(tool="MyTool", email="p.karabowicz@gmail.com")
        results1 = pubmed.query(text, max_results=max1)
        wynik.append(results1)
   #przeksztalcenie wynikow zapytania na data frame
    lista_abstract_3=[]
    for i in wynik:
        for k in i:
            lista_abstract_3.append(k.abstract)

    res = []
    for val in lista_abstract_3:
     if val != None :
      res.append(val)

    list_abstr = ' '.join(res)
    list_abstr = list_abstr.replace('\n', ' ')
    list_abstr = list_abstr.replace("'\'", ' ')
    list_abstr = list_abstr.replace("'", ' ')
    list_analy = list_abstr.lower()
    word = []
    count_word = []
    for w in list_drug_lower:
     count1= list_analy.count(w)
     word.append(w)
     count_word.append(count1)

    df = pd.DataFrame(list(zip(word, count_word)),
                 columns =['Drug name', 'Count'])
    df1 = df[df['Count']>drugs_count1]
    df1.to_csv('drug.csv')
    df1 = df1.sort_values(by=['Count'], ascending=False)
    #st.markdown("<h3 style='text-align: center; color: black;'>Drugs related to protein: </h3>" + df_index['Protein'][0], unsafe_allow_html=True)
    st.write('Drugs related to protein: ' + df_index['Protein'][0])
    df1

    
