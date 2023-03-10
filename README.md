# Patent-Classification
## 디렉토리 소개
* main_project
  * Bert_Pretrain: bert 사전학습 실험 파일들
    * bert_train.py: bert 학습시키는 코드
    * main_pretrained.py: 학습시킨 bert로 분류기 훈련 코드
    * config_pretrained.py: 학습 사용 하이퍼 파라미터
  
  * training_py: 기존 xlm-roberta-base 모델로 분류기 학습
    * model, dataset, datamodule
    * main.py: 분류기 훈련 코드
    * score.ipynb: 점수 확인하는 코드
    
  * DataProcessing_final.ipynb: 사용 데이터 전처리 코드
  
* practice_files: tutorial 연습 파일들
