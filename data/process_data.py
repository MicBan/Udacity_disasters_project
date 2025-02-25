import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function putting together data in csv folder, combining them and returning as pandas dataframe

    Parameters
    ----------
    messages_filepath : string
        Path to csv file with messages
    categories_filepath : string
        Path to csv file with categories

    Returns
    -------
    df : pandas dataframe
        Pandas datafram consising combined data from csv files

    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on = 'id')
    return df

def clean_data(df):
    '''
    Function cleaning the provided dataframe

    Parameters
    ----------
    df : pandas dataframe
        Dataframe to be cleaned

    Returns
    -------
    df : pandas dataframe
        Cleaned dataframe

    '''
    categories = df.categories.str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = list(row.str.split('-',expand=True)[0])
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories',axis=1,inplace=True)
    df = df.join(categories)
    df=df.drop_duplicates()
    df.related[df.related==2] = 1
    return df

def save_data(df, database_filename):
    '''
    Function saving dataframe to the database

    Parameters
    ----------
    df : pandas dataframe
        Dataframe to be saved.
    database_filename : string
        Name of the database.

    Returns
    -------
    None.

    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('messages', engine, index=False,if_exists='replace')  


def main():
    '''
    Function running other functions in a correct order.
    '''
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()