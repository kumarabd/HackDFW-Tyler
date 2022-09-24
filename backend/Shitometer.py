import numpy as np, pandas as pd
import seaborn as sns, matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
#nltk.download('stopwords')
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
def plan1():
    df=pd.read_csv('./news_articles.csv')
    #print(df.head(3))
    df.text = df.title+df.text
    df.drop(columns=["title"], axis = 1, inplace=True)
    #print(df.isnull().sum())
    stop_words = set(stopwords.words('english')) 

    def LemmSentence(sentence):
        lemma_words = []
        wordnet_lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(sentence) 
        for word in word_tokens: 
            if word not in stop_words: 
                new_word = re.sub('[^a-zA-Z]', '',word)
                new_word = new_word.lower()
                new_word = wordnet_lemmatizer.lemmatize(new_word)
                lemma_words.append(new_word)
    #    print(type(" ".join(lemma_words)))
        return " ".join(lemma_words)

    df = df.dropna()
    X = df["text"]
    y = df["label"]
    try:
        X = [LemmSentence(i) for i in X]
    except:
        pass
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    #X = X.dropna()
    print(X.isnull().sum())
    print(y.isnull().sum())
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=7)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # create the transform
    vectorizer = TfidfVectorizer()

    # transforming
    tfidf_train = vectorizer.fit_transform(x_train.iloc[:,0])
    tfidf_test = vectorizer.transform(x_test.iloc[:,0])
    pac = PassiveAggressiveClassifier(random_state = 7,loss = 'squared_hinge',  max_iter = 50, C = 0.16)
    pac.fit(tfidf_train, y_train.values.ravel())

    #Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {round(score*100, 2)}%')
    ax = sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d")
    ax.set(xlabel='Prediction', ylabel='Actual')
    plt.show()

def plan2():
    df=pd.read_csv('./fake.csv')
#   testData = pd.read_csv('./news_articles.csv')
    df.drop(columns=['uuid','ord_in_thread','crawled','thread_title','replies_count','participants_count','likes','comments','shares'], axis=1, inplace=True)
#    print(df.columns)
    df.text = df.title+df.text
    df.drop(columns=["title"], axis = 1, inplace=True)
    #print(df.isnull().sum())
    stop_words = set(stopwords.words('english')) 

    def LemmSentence(sentence):
        lemma_words = []
        wordnet_lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(sentence) 
        for word in word_tokens: 
            if word not in stop_words: 
                new_word = re.sub('[^a-zA-Z]', '',word)
                new_word = new_word.lower()
                new_word = wordnet_lemmatizer.lemmatize(new_word)
                lemma_words.append(new_word)
    #    print(type(" ".join(lemma_words)))
        return " ".join(lemma_words)
    def simplify(values):
        if values == 0:
            return 0
        else:
            return 1
#    def simpleFR(values):
#        if values == 'Fake':
#            return 1
#        else:
#            return 0
    df = df.dropna()
    X = df["text"]
    #print(X.head(10))
    y = df['spam_score']
    y = [simplify(i) for i in y]

#    testData = testData.dropna()
#    X1 = testData["text"]
#    y1 = testData["label"]
#    y1 = [simpleFR(i) for i in y1]
 #   try:
 #       X1 = [LemmSentence(i) for i in X1]
 #   except:
 #       pass
#    print(y[0:100])
#    y = df["label"]
    try:
        X = [LemmSentence(i) for i in X]
    except:
        pass
#    print(x[0:100])
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
#    X1 = pd.DataFrame(X1)
#    y1 = pd.DataFrame(y1)

    #X = X.dropna()
#    print(X.isnull().sum())
#    print(y.isnull().sum())
    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=7)
#    x1_train, x_test, y1_train , y_test = train_test_split(X1, y1, test_size=.99, random_state=7)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # create the transform
    vectorizer = TfidfVectorizer()

    # transforming
    tfidf_train = vectorizer.fit_transform(x_train.iloc[:,0])
    tfidf_test = vectorizer.transform(x_test.iloc[:,0])
    pac = PassiveAggressiveClassifier(random_state = 7,loss = 'squared_hinge',  max_iter = 50, C = 0.16)
    pac.fit(tfidf_train, y_train.values.ravel())

    #Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {round(score*100, 2)}%')
    ax = sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d")
    ax.set(xlabel='Prediction', ylabel='Actual')
    plt.show()
    
def plan3():
    def plan2():
    df=pd.read_csv('./fake.csv')
    testData = pd.read_csv('./news_articles.csv')
    df.drop(columns=['uuid','ord_in_thread','crawled','thread_title','replies_count','participants_count','likes','comments','shares'], axis=1, inplace=True)
#    print(df.columns)
    df.text = df.title+df.text
    df.drop(columns=["title"], axis = 1, inplace=True)
    #print(df.isnull().sum())
    stop_words = set(stopwords.words('english')) 

    def LemmSentence(sentence):
        lemma_words = []
        wordnet_lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(sentence) 
        for word in word_tokens: 
            if word not in stop_words: 
                new_word = re.sub('[^a-zA-Z]', '',word)
                new_word = new_word.lower()
                new_word = wordnet_lemmatizer.lemmatize(new_word)
                lemma_words.append(new_word)
    #    print(type(" ".join(lemma_words)))
        return " ".join(lemma_words)
    def simplify(values):
        if values == 0:
            return 0
        else:
            return 1
    def simpleFR(values):
        if values == 'Fake':
            return 1
        else:
            return 0
    df = df.dropna()
    X = df["text"]
    #print(X.head(10))
    y = df['spam_score']
    y = [simplify(i) for i in y]

    testData = testData.dropna()
    X1 = testData["text"]
    y1 = testData["label"]
    y1 = [simpleFR(i) for i in y1]
    try:
        X1 = [LemmSentence(i) for i in X1]
    except:
        pass
#    print(y[0:100])
#    y = df["label"]
    try:
        X = [LemmSentence(i) for i in X]
    except:
        pass
#    print(x[0:100])
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    X1 = pd.DataFrame(X1)
    y1 = pd.DataFrame(y1)

    #X = X.dropna()
#    print(X.isnull().sum())
#    print(y.isnull().sum())
    x_train, x1_test, y_train, y1_test = train_test_split(X,y, test_size=0.1, random_state=7)
    x1_train, x_test, y1_train , y_test = train_test_split(X1, y1, test_size=.99, random_state=7)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    # create the transform
    vectorizer = TfidfVectorizer()

    # transforming
    tfidf_train = vectorizer.fit_transform(x_train.iloc[:,0])
    tfidf_test = vectorizer.transform(x_test.iloc[:,0])
    pac = PassiveAggressiveClassifier(random_state = 7,loss = 'squared_hinge',  max_iter = 50, C = 0.16)
    pac.fit(tfidf_train, y_train.values.ravel())

    #Predict on the test set and calculate accuracy
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)

    print(f'Accuracy: {round(score*100, 2)}%')
    ax = sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d")
    ax.set(xlabel='Prediction', ylabel='Actual')
    plt.show()
plan2()