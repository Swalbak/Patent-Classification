import numpy as np
import pandas as pd
from UCC_Dataset import UCC_Dataset
import torch
from transformers import AutoTokenizer
from UCC_Data_Module import UCC_Data_Module
from UCC_Comment_Classifier import UCC_Comment_Classifier
import pytorch_lightning as pl


train_path = r'D:\coding\git_repository\RoBERTa\csv_data\ucc_train.csv'
val_path = r'D:\coding\git_repository\RoBERTa\csv_data\ucc_val.csv'

attributes = ['antagonize', 'condescending','dismissive','generalisation',
              'generalisation_unfair','hostile','sarcastic','unhealthy']

model_name = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
ucc_ds = UCC_Dataset(train_path, tokenizer, attributes=attributes)
ucc_ds_val = UCC_Dataset(val_path, tokenizer, attributes=attributes, sample=None)

ucc_data_module = UCC_Data_Module(train_path, val_path, attributes=attributes)
ucc_data_module.setup()
ucc_data_module.train_dataloader()

config = {
    'model_name': 'distilroberta-base',
    'n_labels': len(attributes),
    'batch_size': 128,
    'lr': 1.5e-6,
    'warmup': 0.2, 
    'train_size': len(ucc_data_module.train_dataloader()),
    'weight_decay': 0.001,
    'n_epochs': 100
}

model = UCC_Comment_Classifier(config)
# datamodule
ucc_data_module = UCC_Data_Module(train_path, val_path, attributes=attributes, batch_size=config['batch_size'])
ucc_data_module.setup()

# model
model = UCC_Comment_Classifier(config)

# trainer and fit
# trainer = pl.Trainer(max_epochs=config['n_epochs'], gpus=1, num_sanity_val_steps=50)
trainer.fit(model, ucc_data_module)
