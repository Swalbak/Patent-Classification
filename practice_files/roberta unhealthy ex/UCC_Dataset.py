from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch

class UCC_Dataset(Dataset):

    def __init__(self, data_path, tokenizer, attributes, max_token_len: int=128, sample = 5000):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.max_token_len = max_token_len
        self.sample = sample
        self._prepare_data()

    def _prepare_data(self):
        data = pd.read_csv(self.data_path)
        data['unhealthy'] = np.where(data['healthy'] == 1, 0, 1)

        if self.sample is not None:
            unhealthy = data.loc[data[self.attributes].sum(axis=1) > 0]
            clean = data.loc[data[self.attributes].sum(axis=1) == 0]
            self.data = pd.concat([unhealthy, clean.sample(self.sample, random_state=7)])
        else:
            self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data.iloc[index]
        comment = str(item.comment)
        attributes = torch.FloatTensor(item[self.attributes])
        tokens = self.tokenizer.encode_plus(
            comment,
            add_special_tokens = True,
            return_tensors='pt',
            truncation=True,
            padding='max_length',
            max_length=self.max_token_len,
            return_attention_mask=True
        )

        return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}
