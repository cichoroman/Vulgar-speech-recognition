#!/usr/bin/env python
# coding: utf-8

# In[132]:


# Vulgar Speech

# vulgar words were censored by the names of Latin mushrooms

import requests
from bs4 import BeautifulSoup  
page=requests.get("https://grzyby.pl/atlas-grzybow-przyrodnika.htm")
soup = BeautifulSoup(page.text, 'html.parser')
mushrooms = soup.findAll('span',{'class':'name-latin'})
for i in range (len(mushrooms)):
    print(mushrooms[i].text)


# In[137]:


import csv  
with open('data.csv', 'a+') as csv_file:  
    writer = csv.writer(csv_file)
    for i in range(len(mushrooms)):
        writer.writerow([mushrooms[i].text])

with open('data.csv', 'r') as f:
    read = csv.reader(f, delimiter=";")
    listA = list(read)
    listA = list(filter(None, listA))
print(listA)


# In[142]:


A = ['You are a '+str(listA[0]),'You are my friend','I '+str(listA[11])+' you',
    'Your boyfriend is a '+str(listA[2]),'You are very nice person','I love you',
     'You are a '+str(listA[3]),'You are very fiendly','What the '+str(listA[4]),
     str(listA[5])+' you','Get the '+str(listA[6]),'I like you very much',
    'You are great person','You are such a '+str(listA[10])+' person',
    'Go'+str(listA[8])+' your self','Go clean your room','I '+str(listA[30]),'Just '+str(listA[38]),
    'Just give it back','Just do it','She '+str(listA[99]),'She is the best',str(listA[88])+ ' your '+str(listA[0]),
    str(listA[100])+' you',str(listA[4])+' '+str(listA[200]),
    'Nice pick','Hi you',str(listA[120])+' him']
B = ['negative','positive','negative','negative','positive','positive',
     'negative','positive','negative','negative','negative',
    'positive','positive','negative','negative','positive','negative','negative','positive','positive',
     'negative','positive','negative','negative','negative',
    'positive','positive','negative']
C = [0,1,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,0,1,1,0,1,0,0,0,1,1,0]
D = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z','AB','BC','CD']


# In[143]:


#I created a CSV file consisting of data from these four lists
with open('machinelearningBAZA.csv', 'a+') as csv_file:
    writer = csv.writer(csv_file,delimiter=';')
    for i in range(len(A)):
        writer.writerow ([A[i].strip(),B[i].strip(),C[i],D[i].strip()])
with open('machinelearningBAZA.csv', 'r') as f:
    ml_data = csv.reader(f,delimiter=';')
    lista = list(ml_data)
    lista = list(filter(None, lista))
    print(lista)
    
categories = ['positive', 'negative']


# In[144]:


#I prepared a training dataset by creating 4 separate lists for sentences, categories in words, 
#categories numerically and 'filename' and data entered from the above CSV file.
class Container(object):
     pass 
trening = Container()
trening.target_names=[] 
trening.filenames=[]
trening.data=[]
trening.target=[]
for n in lista:
    trening.data.append(n[0])
    trening.target_names.append(n[1])
    trening.target.append(n[2]) 
    trening.filenames.append(n[3])    

print(trening.data)


# In[145]:


#I used Naive Bayes Algorithm
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(trening.data)
X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, trening.target)


# In[147]:


#sentences to test

docs_new = ['I Boletus edulis you', 'You are very kind person',
'I Boletus reticulatus your apperance','I love you very much'
'We Xerocomus chrysenteron your family','I care a lot', 'Xerocomus chrysenteron this Boletus regius', 'There are so many Leccinum carpini', 'You are my special friend',
'Can we play?','Do you want some?', 'You Xerocomus rubellus' , 'Leccinum aurantiacum your mother','I dont Boletus radicans',
    'You Lepista nebularis at programing ', 'Your boyfriend is a handsome Lepista nebularis ','Just write you Phylloporus ',
       'You are nice girl ','Turn it off ',  'I dont Boletus radicans  ', 'Your boyfriend is a handsome ',
 'Your boyfriend is a handsome ','Go Gymnopus your self', 'They are just Leccinum carpini', 'I Xerocomus rubellus your T-shirt ',
   'You are such a Russula vesca person ','I dont like ', 'You are a bf','I like your Boletus edulis', 'I like your Boletus edulis ',
            'Just give me this Gymnopus peronatus pen!', 'Leccinum a*rantiacum your mother']

X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print (doc + " => "+ trening.target_names[int(category)])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




