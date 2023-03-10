import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from UCC_Dataset import UCC_Dataset

class UCC_Data_Module(pl.LightningDataModule):

  def __init__(self, train_path, val_path, attributes, batch_size: int = 16, max_token_length: int = 128,  model_name='roberta-base'):
    #부모 클래스의 생성자 호출
    super().__init__()
    self.train_path = train_path
    self.val_path = val_path
    self.attributes = attributes
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    #위와 똑같은 방법으로 토크나이저 생성
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  
  def setup(self, stage = None):
    #stage가 None이거나 fit이면
    if stage in (None, "fit"):
      #train, val 데이터 셋을 만든다.
      self.train_dataset = UCC_Dataset(self.train_path, attributes=self.attributes, tokenizer=self.tokenizer)
      self.val_dataset = UCC_Dataset(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)
    #predict면
    if stage == 'predict':
      #검증 데이터 셋만 만든다.
      self.val_dataset = UCC_Dataset(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)

  def train_dataloader(self):
    #train_dataset에 대한 배치를 만들고(미니배치), 매 에포크마다 shuffle하여 overfitting을 방지 -> pytorch의 DataLoader객체
    #num_workers: sub processor, 즉 코어
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True)

  def val_dataloader(self):
    #검증 세트에는 shuffle을 False로
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)
