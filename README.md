# Patent-Classification
## 디렉토리 소개
* main_project
  * Bert_Pretrain: bert 사전학습 실험 파일들(도메인 적응 관련)
    * bert_train.py: bert 학습시키는 코드(도메인 적응)
    * main_pretrained.py: 도메인 적응 bert로 분류기 훈련 코드
    * config_pretrained.py: 학습 사용 하이퍼 파라미터
  
  * training_py: 기존 xlm-roberta-base 모델로 분류기 학습(도메인 적응 X)
    * model, dataset, datamodule
    * main.py: 분류기 훈련 코드
    * score.ipynb: 점수 확인하는 코드
    
  * DataProcessing_final.ipynb: 사용 데이터 전처리 코드
  
* practice_files: tutorial 연습 파일들
* 
**도메인 적응이란?**
  > * pretrain된 모델에 대해, 특정 task에 맞게 모델의 가중치를 직접 학습시킴
  > * 데이터 셋이 적을 때 성능폭 향상이 크기 때문에, 데이터가 충분치 않을 때 사용하는 방법 중 하나
  > * 트랜스포머를 활용한 자연어 처리 9chapter 참고
