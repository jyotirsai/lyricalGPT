import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import random
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

class LyricsDataset(Dataset):
    def __init__(self, data, tokenizer, context_size):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.context_size = context_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        lyric = self.data[idx]
        input_ids = self.tokenizer.encode(lyric).ids
        
        size = len(input_ids)-self.context_size
        random_number = random.randint(0, size - 1)
        x = torch.tensor(input_ids[random_number:random_number+self.context_size],dtype=torch.long)
        y = torch.tensor(input_ids[random_number+1:random_number+self.context_size+1],dtype=torch.long)

        return x, y

def prepare_data(file_path):
    df = pd.read_csv(file_path, header=0)
    data = [df['text'][i] for i in range(len(df))]
    return data

def retrieve_lyric(data):
    for i in range(len(data)):
        yield data[i]
        
def build_tokenizer(config, raw_data):
    tokenizer_path = Path(config["tokenizer_file"])
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["<UNK>", "<PAD>", "<SOS>", "<EOS>"])
        tokenizer.train_from_iterator(
            retrieve_lyric(raw_data), trainer=trainer
        )
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def build_dataloader_and_tokenizers(config):
    text = prepare_data(config['data_file_path'])
    tokenizer = build_tokenizer(config, text)
    new_text = [t for t in text if len(tokenizer.encode(t).ids) <= config['max_token_length'] and len(tokenizer.encode(t).ids) > config['min_token_length']]
    train_size = int(0.9 * len(new_text))
    val_size = len(new_text) - train_size
    raw_train, raw_val = random_split(new_text, [train_size, val_size])

    train = LyricsDataset(
        raw_train,
        tokenizer,
        config["context_size"],
    )
    val = LyricsDataset(
        raw_val,
        tokenizer,
        config["context_size"],
    )

    train_dataloader = DataLoader(train, batch_size=config["batch_size"], shuffle=True)
    val_dataloader = DataLoader(val, batch_size=config["batch_size"], shuffle=True)

    return train_dataloader, val_dataloader, tokenizer