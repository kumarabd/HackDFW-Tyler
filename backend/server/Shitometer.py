import numpy as np, pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def phase1(text):
    df=pd.read_csv('./fake.csv')
    df.drop(columns=['uuid','ord_in_thread','crawled','thread_title','replies_count','participants_count','likes','comments','shares'], axis=1, inplace=True)
    df.text = df.title+df.text
    df.drop(columns=["title"], axis = 1, inplace=True)

    def LemmSentence(sentence, language):
        if language == 'ignore':
            language = 'english'
        stop_words = set(stopwords.words(language))
        lemma_words = []
        wordnet_lemmatizer = WordNetLemmatizer()
        word_tokens = word_tokenize(sentence) 
        for word in word_tokens: 
            if word not in stop_words: 
                new_word = re.sub('[^a-zA-Z]', '',word)
                new_word = new_word.lower()
                new_word = wordnet_lemmatizer.lemmatize(new_word)
                lemma_words.append(new_word)
        return " ".join(lemma_words)
    def simplify(values):
        if values == 0:
            return 0
        else:
            return 1
    df = df.dropna()
    X = df["text"]
    y = df['spam_score']
    y = [simplify(i) for i in y]
    try:
        for i in range(0, len(X)):
            X[i] = LemmSentence(X[i], df[i]['language'])
    except:
        pass
    X = pd.DataFrame(X)
    y = pd.DataFrame(y)

    x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=4)
    testData = pd.DataFrame([LemmSentence(text, 'english')], columns =['text'])
    print(LemmSentence(text, 'english'))
    # The transformation
    vectorizer = TfidfVectorizer()
    tfidf_train = vectorizer.fit_transform(x_train.iloc[:,0])
    tfidf_test = vectorizer.transform(x_test.iloc[:,0])
    pac = PassiveAggressiveClassifier(random_state = 4,loss = 'squared_hinge',  max_iter = 45, C = 0.14)
    pac.fit(tfidf_train, y_train.values.ravel())
    # Predicting
    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    print(score)
    ans = pac.predict(vectorizer.transform(testData.iloc[:,0]))[0]
    if ans == 0:
        return "Real"
    else:
        return "Fake"
 
#text = "why did attorney general loretta lynch plead the fifth barracuda brigade print the administration is blocking congressional probe into cash payments to iran of course she needs to plead the th she either cant recall refuses to answer or just plain deflects the question straight up corruption at its finest  percentfedupcom  talk about covering your ass loretta lynch did just that when she plead the fifth to avoid incriminating herself over payments to irancorrupt to the core attorney general loretta lynch is declining to comply with an investigation by leading members of congress about the obama administrations secret efforts to send iran  billion in cash earlier this year prompting accusations that lynch has pleaded the fifth amendment to avoid incriminating herself over these payments according to lawmakers and communications exclusively obtained by the washington free beacon  sen marco rubio r fla and rep mike pompeo r kan initially presented lynch in october with a series of questions about how the cash payment to iran was approved and delivered  in an oct  response assistant attorney general peter kadzik responded on lynchs behalf refusing to answer the questions and informing the lawmakers that they are barred from publicly disclosing any details about the cash payment which was bound up in a ransom deal aimed at freeing several american hostages from iran  the response from the attorney generals office is unacceptable and provides evidence that lynch has chosen to essentially plead the fifth and refuse to respond to inquiries regarding herrole in providing cash to the worlds foremost state sponsor of terrorism rubio and pompeo wrote on friday in a followup letter to lynch more related"   
#print(phase1(text))
