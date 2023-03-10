import os
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
import torch
import pytorch_lightning as pl

class Patent_Classifier(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict=True)
        self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
        
        # initialize
        torch.nn.init.xavier_uniform_(self.hidden.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)

        self.loss_func = nn.BCEWithLogitsLoss()
        self.dropout = nn.Dropout()
    
    def forward(self, input_ids, attention_mask, labels=None):
        # 토큰별로 각각 768차원의 데이터?
        output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        # 토큰별로 평균낸 768개 데이터?
        pooled_output = torch.mean(output.last_hidden_state, 1)

        pooled_output = self.dropout(pooled_output)
        pooled_output = self.hidden(pooled_output)
        pooled_output = F.relu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = 0
        if labels is not None:
            # loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
            loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels)

        return loss, logits
    
    def training_step(self, batch, batch_index):
        # forword(input ids, attention_mask)에 batch들을 넘겨줌(model 인스턴스 호출 -> forword 메소드 호출)
        loss, outputs = self(**batch)
        self.log("train loss ", loss, prog_bar = True, logger=True)

        return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}

    def validation_step(self, batch, batch_index):
        loss, outputs = self(**batch)
        self.log("validation loss ", loss, prog_bar = True, logger=True)

        return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}

    def predict_step(self, batch, batch_index):
        loss, outputs = self(**batch)

        return {'loss': loss, 'outputs': outputs}
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
        total_steps = self.config['train_size']/self.config['batch_size']
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        warmup_steps = math.floor(total_steps * self.config['warmup'])
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        
        return [optimizer],[scheduler]