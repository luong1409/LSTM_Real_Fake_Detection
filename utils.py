import pandas as pd
from sklearn.model_selection import train_test_split
import os

def trim_string(raw:str, first_n_words=200):
    """This is the function for truncate the input sentence.

    Args:
        raw (str): input sentence
        first_n_words (int): the length of output sentence.

    Returns:
        output (str): the string after be truncated.
    """
    raw = raw.split(maxsplit=first_n_words)
    output = ' '.join(raw[:first_n_words])
    return output

def preprocess_dataset(dataset_path:str, train_test_ratio = 0.1, train_valid_ratio = 0.8, save_path="data"):
    """Proprocess (trimming long text) and split to train, test and valid dataset.

    Args:
        dataset_path (str): the path to raw dataset.
        train_test_ratio (float, optional): the ratio for splitting train and test dataset. Defaults to 0.1.
        train_valid_ratio (float, optional): the ratio for splitting train and valid dataset. Defaults to 0.8.
        save_path (str): the path to folder for saving dataset. Defaults to "data".
    Ouput:
        Train, test, valid dataset save in save_path.
    """
    
    df_raw = pd.read_csv(filepath_or_buffer=dataset_path)
    
    df_raw['label'] = (df_raw['label'] == 'FAKE').astype('int')
    df_raw['titletext'] = df_raw['title'] + ". " + df_raw["text"]
    df_raw = df_raw.reindex(columns=["label", 'title', 'text', 'titletext'])
    
    df_raw.drop(df_raw[df_raw.text.str.len() < 5].index, inplace=True, axis='index')
    
    df_raw['text'] = df_raw['text'].apply(trim_string)
    df_raw['titletext'] = df_raw['titletext'].apply(trim_string) 
    
    df_real = df_raw[df_raw['label'] == 0]
    df_fake = df_raw[df_raw['label'] == 1]
    
    # train, test split
    df_real_train_full, df_real_test = train_test_split(df_real, test_size=train_test_ratio, random_state=1)
    df_fake_train_full, df_fake_test = train_test_split(df_fake, test_size=train_test_ratio, random_state=1)
    
    # train, valid split
    df_real_train, df_real_valid = train_test_split(df_real_train_full, test_size=train_valid_ratio, random_state=1)
    df_fake_train, df_fake_valid = train_test_split(df_fake_train_full, test_size=train_valid_ratio, random_state=1)
    
    df_train = pd.concat(objs=[df_real_train, df_fake_train], ignore_index=True, sort=False)
    df_test = pd.concat(objs=[df_real_test, df_fake_test], ignore_index=True, sort=False)
    df_valid = pd.concat(objs=[df_real_valid, df_fake_valid], ignore_index=True, sort=False)
    
    if save_path == None:
        save_path = "data"
    
    os.makedirs(name=save_path, exist_ok=True)
    df_train.to_csv('data/train.csv', index=False)
    df_test.to_csv('data/test.csv', index=False)
    df_valid.to_csv('data/valid.csv', index=False)