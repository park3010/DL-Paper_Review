# Sequence to Sequence Learning with Neural Networks

<br>

## Abstract

<br>

- 기존 심층 신경망(이하 DDNs)는 대규모의 라벨링된 훈련 데이터가 있어야 하며, 시퀸스를 시퀀스로 매핑하지 못함
- 즉, language sentence의 토큰 하나 하나를 입력값을 보고 입력된 벡터와 동일한 개수의 출력값을 뱉어냄
  - "나는 너를 사랑해" 라는 한국어를 영어로 번역하는 경우 동일한 개수의 토큰을 가진 "I love you"로 번역함
  - "나는 너를 정말 사랑해" 를 영어로 번역 시 "I love you so much"로 나와야 자연스러우나 입력한 토큰 개수에 맞게 출력되므로 "I love you very"와 같이 어색한 문장이 출력될 수 있음

- but 본 논문에서 소개하는 Seq2Seq 모델은 개별 토큰이 아닌 문장 전체(시퀸스)를 한 단위로 보기 때문에 입력 시퀸스와 출력 시퀀스의 토큰 개수가 달라도 됨
- 다층 모델(LSTM)을 사용하여 입력 시퀀스를 고정된 차원의 벡터로 매핑한 뒤 또 다른 다층 LSTM을 이용하여 해당 벡터에서 목표 시퀀스로 디코딩함


<br>

## Introduction

<br>

- DNN은 음성 인식, 객체 인식 등 분야에서 뛰어난 성능을 보이나 input과 target vector가 고정되어(fixed) 있어 sequential problem을 제대로 해결할 수 없다는 한계점이 존재함
- 본 논문에서는 두 개의 LSTM을 encoder와 decoder로 사용하여 입력 시퀀스를 한 번에 한 타임스텝싹 읽어서 고정된 차원의 큰 벡터 표현을 얻고 해당 벡터에서 출력 시퀀스를 추출함
  ![image](https://github.com/user-attachments/assets/f4f775fb-0e68-4e7e-a419-97f105e4d811)
  ```
  두 번째 LSTM은 입력 시퀀스에 의해 조건이 부여된 순환 신경망(RNN) 기반의 언어 모델과 유사
  ```

  <br>

- LSTM은 긴 문장에 대해서도 성능 유지함, 이는 학습 데이터의 원본(source) 문장의 단어 순서는 뒤집고 target 문장의 단어 순서는 그대로 유자하여 원문과 번역문 사이의 많은 단기 종속성(short-term dependencies)이 형성되어 최적화 문제가 단순화되었기 때문


<br>

## The model

<br>

- 


