import pandas as pd
import numpy as np
import string

import nltk
try:
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
except:
    nltk.download('punkt')
    nltk.download('stopwords')

    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score




def read_data(url: str):
    print(f"reading data from {url}")
    data = pd.read_csv(url)
    return data


def count_total_words(text):
    char = 0
    for word in text.split():
        char += len(word)
    return char

def convert_lowercase(text):
    text = text.lower()
    return text

def remove_punc(text):
    exclude = string.punctuation
    return text.translate(str.maketrans('', '', exclude))

def remove_stopwords(text):
    new_list = []
    words = word_tokenize(text)
    stopwrds = stopwords.words('english')
    for word in words:
        if word not in stopwrds:
            new_list.append(word)
    return ' '.join(new_list)

def perform_stemming(text):
    stemmer = PorterStemmer()
    new_list = []
    words = word_tokenize(text)
    for word in words:
        new_list.append(stemmer.stem(word))

    return " ".join(new_list)



def feature_engineering(data: pd.DataFrame):
    print("Feature Engineering")
    data['Total Words'] = data['Message'].apply(lambda x: len(x.split()))
    data['Total Chars'] = data["Message"].apply(count_total_words)
    
    data['Message'] = data['Message'].apply(convert_lowercase)
    data['Message'] = data['Message'].apply(remove_punc)
    data['Message'] = data['Message'].apply(remove_stopwords)

    data['Message'] = data['Message'].apply(perform_stemming)
    data['Total Words After Transformation'] = data['Message'].apply(lambda x: np.log(len(x.split())))

    data['Category'] = data['Category'].replace({'spam':0,'ham':1})

    return data


def text_vectorization(data):
    print("Text Vectorization")
    X = data["Message"]
    y = data['Category'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42, stratify = y)
    tfidf = TfidfVectorizer(max_features= 2500, min_df= 2)
    X_train = tfidf.fit_transform(X_train).toarray()
    X_test = tfidf.transform(X_test)

    return X_train, X_test, y_train, y_test, tfidf


def train_model(X_train, X_test, y_train, y_test):
    print("Training Model")
    model = RandomForestClassifier(n_estimators= 300)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    
    accuracy = round(accuracy_score(y_test, y_pred), 3)
    precision = round(precision_score(y_test, y_pred), 3)
    recall = round(recall_score(y_test, y_pred), 3)

    print(f'Accuracy of the model: {accuracy}')
    print(f'Precision Score of the model: {precision}')
    print(f'Recall Score of the model: {recall}')




if __name__ == "__main__":
    data = read_data('data/SPAM_text_message.csv')
    data = feature_engineering(data)
    X_train, X_test, y_train, y_test, tfidf = text_vectorization(data)
    train_model(X_train, X_test, y_train, y_test)