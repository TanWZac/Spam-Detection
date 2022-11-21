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


def lemm(text):
    repunc = re.sub(clear, ' ', text)
    nonstop = [j for j in repunc if j not in set(stopwords.words('english'))]
    lem = WordNetLemmatizer()
    return "".join([lem.lemmatize(j) for j in nonstop])


# In[9]:


df['MESSAGE'] = df['MESSAGE'].apply(lemm)


# In[10]:


corpus = df['MESSAGE'].tolist()


# In[11]:


df['MESSAGE']


# In[27]:


### this step is to save the dataframe into csv file as the above algorithm process too long
df.to_csv("NLP.csv", encoding='utf-8')


# In[28]:


df = pd.read_csv("NLP.csv")


# In[30]:


df = df.iloc[:, 1:]


# In[31]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(ngram_range=(1, 3))
X = tfidf.fit_transform(corpus)
y = df["CATEGORY"]


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[14]:


import seaborn as sns


# In[15]:


import xgboost as xg
classifierXg = xg.XGBClassifier()
classifierXg.fit(X_train, y_train)


# In[16]:


import lightgbm as lgb
classifier = lgb.LGBMClassifier()
classifier.fit(X_train, y_train)


# In[18]:


from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train, y_train)


# In[19]:


from sklearn import tree
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier
classifierKNN = KNeighborsClassifier(n_neighbors=2)
classifierKNN.fit(X_train, y_train)


# In[32]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 90)
rf.fit(X_train, y_train)


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


# In[21]:


report("XGBoost", X_test, y_test, classifierXg)


# In[22]:


report("lightgbm", X_test, y_test, classifier)


# In[23]:


report("KNN", X_test, y_test, classifierKNN)


# In[24]:


report("Naive Bayes", X_test, y_test, NB)


# In[25]:


report("Decision Tree", X_test, y_test, dt)


# In[33]:


report("Random Forest", X_test, y_test, rf)

