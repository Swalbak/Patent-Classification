import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

class Patent_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, labels, max_token_len :int=256):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_token_len = max_token_len

        self._prepare_data()
    
    def _prepare_data(self):
        self.data = pd.read_csv(self.data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        abstract = item['Abstract (Original Language)']
        labels = torch.FloatTensor(item[self.labels])
        tokens = self.tokenizer.encode_plus(abstract,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=self.max_token_len,
                                            padding='max_length',
                                            return_attention_mask=True
                                            )
        
        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': labels}
