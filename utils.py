import pandas as pd
from sklearn.model_selection import train_test_split
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence



class NewsDataset(Dataset):
    """Initialize

    Args:
        df (pd.DataFrame): train or test or valid dataframe.
    """
    def __init__(self, df:pd.DataFrame):
        self.labels = df['label'].values
        self.texts = df['titletext'].values
        
    def __len__(self):
        return len(self.labels)
    
    def get_batch_labels(self, index):
        return self.labels[index]
    
    def get_batch_texts(self, index):
        return self.texts[index]
    
    def __getitem__(self, index):
        batch_texts = self.get_batch_texts(index)
        batch_labels = self.get_batch_labels(index)
        
        return batch_labels, batch_texts

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
    
    
    
def yield_tokens(data_iter, tokenizer):
    """generate token from input text

    Args:
        data_iter (Iterator): the Iterator object through all text in dataset.
        tokenizer (): the tokenizer for tokenize all text included in dataset.

    Yields:
        The list of tokens contains tokenized text
    """
    for text in data_iter:
        yield tokenizer(text)
    
    
def generate_dataset(path:str):
    """Generate dataset from file .csv

    Args:
        path (str): the path to file .csv

    Returns:
        dataset (NewsDataset): dataset create from file .csv
    """
    df = pd.read_csv(path)
    df['text_len'] = df.apply(lambda x:len(x.text), axis=1)
    df = df.sort_values(by="text_len")
    
    dataset = NewsDataset(df)
    
    return dataset


def get_vocab_from(df_path:str="data/train.csv"):
    """Generate vocabulary from all texts of file .csv

    Args:
        df_path (str, optional): The path to file .csv. Defaults to "data/train.csv".

    Returns:
        Vocabulary: The vocabulary includes all words that have been tokenized from dataset.
    """
    df = pd.read_csv(df_path)
    tokenizer = get_tokenizer(tokenizer="basic_english")
    
    vocab = build_vocab_from_iterator(
        yield_tokens(iter(df.titletext), tokenizer=tokenizer), 
        specials=["<pad>", "<unk>"],
        min_freq=3,
    )
    
    vocab.set_default_index(index=vocab['<unk>'])
    
    return vocab


def get_dataloader(path:str):
    """Generate dataloader from file .csv

    Args:
        path (str): the path to file .csv

    Returns:
        DataLoader: 
    """
    
    vocab = get_vocab_from()
    tokenizer = get_tokenizer(tokenizer="basic_english")

    def collate_fn(batch):
        text_pipeline = lambda x: vocab(tokenizer(x))
        label_list, text_list, text_len = [], [], []
        
        for (label_, text_) in batch:
            label_list.append(label_)
            preprocessed_text = text_pipeline(text_)
            text_list.append(torch.LongTensor(preprocessed_text))
            text_len.append(len(preprocessed_text))
        
        padded_sequences = pad_sequence(sequences=text_list, padding_value=vocab["<pad>"], batch_first=True)
        
        return torch.Tensor(label_list), padded_sequences, text_len
    
    dataset = generate_dataset(path=path)
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return data_loader