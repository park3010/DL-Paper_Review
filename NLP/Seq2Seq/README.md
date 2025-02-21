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
- 본 논문에서는 두 개의 LSTM을 encoder와 decoder로 사용하여 입력 시퀀스를 한 번에 한 타임스텝싹 읽어서 고정된 차원의 큰 벡터 표현을 얻고 해당 벡터에서 출력 시퀀스를 추출함 <br>
  ![image](https://github.com/user-attachments/assets/f4f775fb-0e68-4e7e-a419-97f105e4d811)
  ```
  두 번째 LSTM은 입력 시퀀스에 의해 조건이 부여된 순환 신경망(RNN) 기반의 언어 모델과 유사
  ```

  <br>

- LSTM은 긴 문장에 대해서도 성능 유지함, 이는 학습 데이터의 원본(source) 문장의 단어 순서는 뒤집고 target 문장의 단어 순서는 그대로 유자하여 원문과 번역문 사이의 많은 단기 종속성(short-term dependencies)이 형성되어 최적화 문제가 단순화되었기 때문


<br>

## The model

<br>

- RNN은 기본적으로 순차 데이터를 다룰 수 있도록 전방향 신경망을 일반화한 것으로 sequencial problem에 매우 적합한 model임 <br>
  ![image](https://github.com/user-attachments/assets/a099cc85-4ddd-4c10-a847-4f86dc8f8b55)
- but 입력 시퀀스와 출력 시퀀스의 길이가 다를 경우 좋은 성능일 보이기 어려움, 또한 RNN은 장기 의존성(long-term dependencies)이 발생할 수 있음
- 본 논문에서는 LSTM을 이용하여 장기 의존성을 해결하고자 함

<br>

#### 기존 LSTM

  ![image](https://github.com/user-attachments/assets/eb53538b-3fdf-4480-a979-2baa32f3f9fa)
  ```
   LSTM의 목표는 조건부 확률 $p(y_1, ..., y_{T^'}|x_1, ..., x_T)$ 를 추정하는 것

  - $(x_1, ..., x_T)$ 는 입력 시퀀스, $(y_1, ..., y_{T^'})$ 출력 시퀀스
  - 출력 시퀀스 길이 $T^'$와 입력 시퀀스 길이 $T$ 는 다를 수 있음

  - LSTM은 입력 시퀀스 $(x_1, ..., x_T)$의 마지막 hiddent state로부터 고정된 차원의 표현 $v$를 얻음
  - LSTM-LM을 사용하여 $v$로부터 $y_1, ..., y_{T^'}$의 확률을 계산함

  - 어휘(Vocabulary) 내 모든 단어에 대한 softmax를 표현하여 $p(y_t|v, y_1, ..., y_{t-1})$ 분포를 구함
  ```

<br>

#### 본 논문의 LSTM 차이점
  - 입력 시퀀스를 위한 LSTM, 출력 시퀀스를 위한 LSTM 두 개를 사용하여 모델의 parameter 수는 증가했지만 연산 비용은 거의 증가하지 않음
  - 4개의 layer로 구성된 심층 LSTM 사용
  - 입력 문장의 단어 순서를 뒤집음  <br>
    -> 원본 문장이 "a, b, c"라면 기존에는 "α, β, γ"로 매핑했으나 본 논문에서는 "c, b, a"를 "α, β, γ"로 매핑하여 SGD를 통해 입출력 간의 연결을 쉽게 설정할 수 있도록 함 <br>

  ![image](https://github.com/user-attachments/assets/f4024aaf-1ed1-4faa-9443-a8e78c919bf5)
  ```
   입력 시퀀스 내 토큰
  <sos> : "start of sequence", 시퀀스가 시작됨을 나타냄
  <eos> : "end of sequence", 시퀀스가 끝났음을 나타냄

   [encoder]
  - 
  ```


<br>

## Experiments

<br>

- 




