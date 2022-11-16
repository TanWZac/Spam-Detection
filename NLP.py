#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv("C:/Users/twz18/Downloads/Spam Email raw text for NLP.csv")
df


# In[2]:


df.head(50)


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


# In[7]:


import string
import re
clear = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


# In[8]:


lem = WordNetLemmatizer()
corpus = []

def process(msg):
    for i in range(len(df)):
        repunc = re.sub(clear, ' ', msg[i])
        nonstop = [j for j in repunc if j not in set(stopwords.words('english'))]
        cor = [lem.lemmatize(j) for j in nonstop]
        cor = "".join(cor)
        corpus.append(cor)
    


# In[9]:


process(df['MESSAGE'])


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3))
X = tfidf.fit_transform(corpus)
y = df["CATEGORY"]


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[16]:


import seaborn as sns


# In[17]:


import xgboost as xg
classifierXg = xg.XGBClassifier()
classifierXg.fit(X_train, y_train)


# In[18]:


y_pred = classifierXg.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = accuracy_score(y_pred, y_test)
report = classification_report(y_test, y_pred)
print(report)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print("Accuracy of LightGBM Model:", acc*100,"%")


# In[19]:


import lightgbm as lgb
classifier = lgb.LGBMClassifier()
classifier.fit(X_train, y_train)


# In[20]:


y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
acc = accuracy_score(y_pred, y_test)
report = classification_report(y_test, y_pred)
print(report)
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True)
print("Accuracy of LightGBM Model:", acc*100,"%")

