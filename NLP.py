#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("C:/Users/twz18/Downloads/Spam Email raw text for NLP.csv")
df


# In[2]:


df.head()


# In[3]:


def replacement(df, col):
    df[col] = df[col].str.replace(r'<[^<>]*>', '', regex=True)
    df[col] = df[col].str.replace(r'http', '', regex=True)
    df[col] = df[col].str.replace(r'[^A-Za-z0-9(),!?@\'\`\"\_\n]', '', regex=True)
    df[col] = df[col].str.replace(r'\n', '', regex=True)
    df[col] = df[col].str.lower()
    return df


# In[4]:


df = replacement(df, 'MESSAGE')


# df['MESSAGE'] = df['MESSAGE'].str.replace(r'<[^<>]*>', '', regex=True)

# In[5]:


df.info()


# In[6]:


import nltk
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


# In[8]:


import string
import re
clear = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


# In[9]:


lem = WordNetLemmatizer()
corpus = []

def process(msg):
    for i in range(len(df)):
        repunc = re.sub(clear, ' ', msg[i])
        nonstop = [j for j in repunc if j not in set(stopwords.words('english'))]
        cor = [lem.lemmatize(j) for j in nonstop]
        cor = "".join(cor)
        corpus.append(cor)
    


# In[10]:


process(df['MESSAGE'])


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3))
X = tfidf.fit_transform(corpus)
y = df["CATEGORY"]


# In[72]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[61]:


import seaborn as sns


# In[62]:


import xgboost as xg
classifierXg = xg.XGBClassifier()
classifierXg.fit(X_train, y_train)


# In[63]:


import lightgbm as lgb
classifier = lgb.LGBMClassifier()
classifier.fit(X_train, y_train)


# In[64]:


from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors=2)
classifierKNN.fit(X_train, y_train)


# In[87]:


from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, y_train)


# In[20]:


def report(model_name, X_test, y_test, model):
    y_pred = model.predict(X_test)
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc = accuracy_score(y_pred, y_test)
    report = classification_report(y_test, y_pred)
    print(report)
    cm = confusion_matrix(y_pred, y_test)
    sns.heatmap(cm, annot=True)
    print("Accuracy of " + model_name + " Model:", acc*100,"%")


# In[22]:


report("XGBoost", X_test, y_test, classifierXg)


# In[23]:


report("lightgbm", X_test, y_test, classifier)


# In[29]:


report("KNN", X_test, y_test, classifierKNN)


# In[88]:


report("Naive Bayes", X_test, y_test, NB)

