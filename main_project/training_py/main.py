import os
from datamodule import Patent_Data_Module
from config import config
import pickle
from model import Patent_Classifier
import pytorch_lightning as pl
import torch
import argparse

# 여기만 바꾸기
# 2, 3, 4
sliced_num = 4

# 20000, 200000, 400000, not
sample_num = 20000

# 
epoch_num = config['n_epochs']

print(f"{sample_num}Sample, {epoch_num}epoch train started!")

data_path = f"/home/aailab/yjh/csv_data/patent_data/edited_data/BCELOSS/{sliced_num}sliced/{sample_num}Sample/"
pred_path = f"/home/aailab/yjh/result/multi_label_project/BCELOSS/{sliced_num}sliced/{epoch_num}epoch/"

with open(data_path + 'labels.pickle', 'rb') as fr:
    labels = pickle.load(fr)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

train_path = data_path + 'train_data.csv'
val_path = data_path + 'val_data.csv'
test_path = data_path + 'test_data.csv'

config['n_labels'] = len(labels)

data_module = Patent_Data_Module(train_path, val_path, labels=labels, batch_size=config['batch_size'], model_name=config['model_name'])
data_module.setup()
config['train_size'] = len(data_module.train_dataloader())
model = Patent_Classifier(config)

trainer = pl.Trainer(max_epochs=config['n_epochs'], num_sanity_val_steps=-1, accelerator="gpu", devices=4, strategy="dp")
# trainer = pl.Trainer(max_epochs=config['n_epochs'], accelerator="gpu", devices=4, strategy="dp")
trainer.fit(model, data_module)

predictions = trainer.predict(model, data_module)
torch.save(model.state_dict(), pred_path+f'{sample_num}SampleModel.pth')

with open(pred_path+f'{sample_num}SamplePredictions.pickle', 'wb') as fw:
    pickle.dump(predictions, fw)

print(f"{sample_num}Sample, {epoch_num}epoch train completed!")
