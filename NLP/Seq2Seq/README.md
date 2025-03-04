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
  - embedding layer을 통과한 첫 번째 입력 토큰 <sos>와 초기 은닉값 h0을 인코더에 함께 통과시키면 h1이 출력됨
  - 출력된 h1을 embedding layer을 통과한 두 번째 토큰인 "guten"의 임베딩 값과 함께 다음 인코더에 함께 통과시키면 h2가 출력됨
  - 이 과정을 반복하며, 마지막 인코더에서 출력되는 h4는 고정된 크기의 context vector(v)가 됨

   [dncoder]
  - 인코더에서 출력된 context vector는 embedding layer을 통과한 <sos> 토큰과 함께 디코더를 통과시켜 s1을 출력함
  - s1는 Linear function을 거쳐 "good"을 나타내는 벡터로 출력됨
  - 해당 벡터 출력값 y1은 s1과 함께 다음 디코더로 입력되어 s2를 출력하게 됨
  - 출력한 s2은 동일하게 Linear function을 거쳐 y2를 출력하며, s2와 y2는 다음 디코더의 입력으로 들어가게 됨
  - <eos>가 출력될 때까지 이 과정을 반복함
  ```


<br>

## Experiments

<br>

- 본 논문에서는 WMT’14의 English to French dataset으로 실험을 진행함
  - source / target language 각각에 고정된(fixed) size vocabulary를 사용함(source-160,000개 / target-80,000개)
  - vocabulary에 포함되지 않은 모든 단어는 "UNK" 토큰으로 대체됨

<br>

- 가장 가능성 높은 번역을 찾기 위해 단순히 좌->우로 진행하는 빔 서치(beam search) 디코더를 사용
  ![image](https://github.com/user-attachments/assets/94c6b565-4e87-4901-a296-712f8e1dc5b6)

<br>

- LSTM 자체도 장기 의존성 문제를 해결할 수 있으나 source 문장의 순서를 뒤집었을 경우(LSTM에서 target 문장은 뒤집지 않음) LSTM이 훨씬 더 잘 학습함을 발견함 <br>
  -> LSTM의 test perplexity는 5.8에서 4.7로 감소, 디코딩된 번역의 text BLEU 점수는 25.9에서 30.6으로 증가함
  ```
  - 순서대로 source 문장과 target 문장을 연결할 경우 연결되는 단어쌍 사이의 거리는 모두 동일함
  - but 순서를 뒤집을 경우 문장의 앞에 위치한 단어일수록 target 문장의 단어 간 거리가 짧아지게 되나 source 문장의 뒤에 위치한 단어들에 대해선 순서을 뒤집었을 경우 오히려 단어쌍 사이의 거리가 더 멀어짐
  - 순서를 그대로 유지하거나 뒤집었을때나 결국 단어쌍 사이의 거리 평균값은 동일함
  => sequencial problem에서 문장에 앞에 위치한 값이 모든 문장의 값에 영향을 주기 때문에 앞 쪽에 위치한 값일수록 중요도가 높다고 보므로 순서를 뒤집을 경우 더 중요도 높은 값에 대해 더 좋은 성능을 보장하는 효과를 보이게 됨
  ```

<br>

##  Training details

<br>

- 구조: 4개의 layer를 가진 심층 LSTM
  - 각 layer 별 1000개의 cell
  - 1000 차원의 word embedding
  - 입력 vocab size : 160,000
  - 출력 vocab size : 80,000

- learning
  - SGD
  - learning rate = 0.7 -> 5 epoch 이후 learning rate를 0.5배씩 적용
  - batch size = 128
  - exploding gradient 문제 발생하므로 강한 제약을 준 gradient 사용
 
<br>

![image](https://github.com/user-attachments/assets/ead8ede4-6956-45eb-ba6b-f96f5108029e)
