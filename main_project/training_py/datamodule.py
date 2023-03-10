from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import Patent_Dataset

class Patent_Data_Module(pl.LightningDataModule):
    def __init__(self, train_path, val_path, labels, batch_size: int=16, max_token_len: int=256, model_name='xlm-roberta-base'):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.labels = labels
        self.batch_size = batch_size
        self.max_token_len = max_token_len
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = Patent_Dataset(self.train_path, self.tokenizer, self.labels)
            self.val_dataset = Patent_Dataset(self.val_path, self.tokenizer, self.labels)
        if stage == 'predict':
            self.val_dataset = Patent_Dataset(self.val_path, self.tokenizer, self.labels)
        # if stage == 'test':
        #     self.test_dataset = Patent_Dataset(self.)
            
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)
    
    def predict_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)