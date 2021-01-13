import sys
import re
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

import pickle
from sqlalchemy import create_engine


nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    '''
    Loading data from the SQL Lite database and returning X, Y and column names

    Parameters
    ----------
    database_filepath : string
        path to the database

    Returns
    -------
    X : pandas series
        series consisting of X values
    Y : pandas df
        dataframe consisting of X values
    cat_names : list
        list of category names

    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from messages', con=engine)
    X = df.message
    Y = df[df.columns[4:]]
    cat_names = df.columns[4:]
    return X, Y, cat_names

def tokenize(text):
    '''
    Function cleaning provided text by replacing URLs with dummy name, removing stop words, tokenising and lemmatizing it 

    Parameters
    ----------
    text : string
        any string requiring cleaning

    Returns
    -------
    clean_tokens : list
        list of tokens

    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return clean_tokens


def build_model():
    '''
    Building a model with a help of SciKit pipeline

    Returns
    -------
    function
        grid search function helping to choose best parameters for the model

    '''
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier())),])
    
    parameters = {
    'tfidf__smooth_idf': (True,False),
    'clf__estimator__criterion': ('gini', 'entropy')}
    
    return GridSearchCV(pipeline, param_grid=parameters)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function evaluating results of the model

    Parameters
    ----------
    model : function
        SciKit model
    X_test : pandas series
        series of test X values
    Y_test : pandas dataframe
        series of test Y values
    category_names : list
        category names

    Returns
    -------
    None.

    '''
    y_pred = pd.DataFrame(model.predict(X_test),columns=category_names)
    print('Classification report for all categories:')
    print(classification_report(Y_test,y_pred))
    # print('Classification reports separate for each category:')
    # for column in category_names:
    #     print(f'Classification report for {column} category')
    #     print(classification_report(Y_test[column],y_pred[column]))
    


def save_model(model, model_filepath):  
    '''
    Function saving model as a pickle file

    Parameters
    ----------
    model : function
        ML model to be saved
    model_filepath : string
        Path under which model is to be saved

    Returns
    -------
    None.

    '''
    pickle.dump(model, open(model_filepath,'wb'))


def main():
    '''
    Main function running other functions in right order

    Returns
    -------
    None.

    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()