import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from transformers import AutoModelForMaskedLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0, 1, 2, 3"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

# 훈련 epoch 개수
# 여기만 변경!!!!!!!!!
epoch_num = 10

save_dir = f'/home/aailab/yjh/src/multi label project/python_file/save_model_{epoch_num}epoch/'

os.makedirs(save_dir, exist_ok=True)

ds = pd.read_csv("/home/aailab/yjh/csv_data/patent_data/edited_data/BCELOSS/4sliced/bert_pretrain.csv", index_col=0)

ds = ds['Abstract (Original Language)']

model_ckpt = 'xlm-roberta-base'
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(batch):
    return tokenizer(batch, truncation=True, max_length=512, return_special_tokens_mask=True)

ds_mlm = ds['Abstract (Original Language)'].map(tokenize)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
data_collator.return_tensors = 'pt'

train_set, eval_set = train_test_split(ds_mlm, test_size=0.1, random_state=42)

training_args = TrainingArguments(
    output_dir = save_dir,
    num_train_epochs=epoch_num,
    per_device_train_batch_size=8,
    logging_strategy='steps',
    evaluation_strategy='epoch',
    save_strategy='no',
    # log_level='error',
    report_to='none',
    push_to_hub=False
    )

# args
trainer = Trainer(
    model = AutoModelForMaskedLM.from_pretrained(model_ckpt),
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_set,
    eval_dataset=eval_set    
)

print("Train start!")

trainer.train()
print("bert train completed!!")

trainer.save_model()
print(f"bert save completed at {save_dir}!!")