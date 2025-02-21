# A Neural Probabilistic Language Model 

<br>

## Abstract

<br>

- 기존 통계적 언어 모델의 목표는 단어 시퀀스(단어들 간의 순서)의 결합 확률 함수(joint probability function)를 학습하는 것 <br>
  -> but "차원의 저주"로 인한 어려움 존재

- 본 논문에서는 각 단어에 대한 분산 표현(각 단어들 간의 유사성)을 학습함과 동시에 단어 시퀀스의 확률 함수를 학습하는 식으로 "차원의 저주"를 해결하고자 함

  <br>

  #### 차원의 저주

  ![image](https://github.com/user-attachments/assets/49b50e66-6d6d-4eb5-95e1-2edbbec9a4ea)

  ```
  - 학습을 위해 차원이 증가함에 따라(=변수의 수 증가) 학습 데이터보다 차원의 수가 많아지면서 성능이 저하하는 현상
  - 개별 차원 내 학습할 데이터 수가 적어지는 현상
  ```

<br>

## Introduction

<br>

### 차원의 저주

- 이산 확률 변수간의 결합 분포 모델링 시 '차원의 저주'로 인한 학습 문제 어려움이 발생함
- 단어 수가 100,000개인 자연어에서 10개의 연속된 단어의 결합 분포 모델링 시 가능한 free parameters는 $100,000^10 - 1 = 10^50 - 1$ 개
 
<br>

- 통계적 Language model에서 단어 시퀀스에서 이전 단어들이 주어졌을때 다음 단어가 나타날 조건부 확률로 표현함
  ![image](https://github.com/user-attachments/assets/9428feb7-90da-4f52-b839-cdae58d6231a)
  - $w_t$ 는 $t$번째 단어, 부분 시퀀스는 $w^i_t = (w_i, w_{i+1}, ..., w_{i-1}, w_i)$로 표현
 
<br>

- 기존의 통계적 모델은 문장에서 가까운 단어들 간의 통계적 의존성을 가진다고 고려하여 문제의 난이도를 줄이며 대표적인 방법이 **N-gram 모델** 임
  - 방대한 수의 문맥(context) 각각에 대해 다음 단어가 등장할 확률을 나타내는 표를 생성한 뒤 마지막 $n−1$ 개의 단어 조합을 활용하여 다음 단어의 확률을 근사하는 것
  
    ![image](https://github.com/user-attachments/assets/d774ecb6-c087-40b7-8f6c-3f6aa7246215)
  
- but 이러한 방법은 training corpus에서 자주 등장하는 단어 조합만 고려하며 등장하지 않는 새로운 n개의 단어 조합에 대해선?
- 더 짧은 context을 사용하여 일반화(generalization)함 -> 긴 단어 시퀀스에 대해서는 부분 시퀀스를 이어붙이는 방식 사용
  ```
  두 가지 문제 존재
  1) 1 ~ 2개의 단어만 고려, 더 먼 context 고려 안함
  2) 단어 간 유사성 고려 안 함
  ```


<br>

### Fighting the Curse of Dimensionality with its Own Weapons

- 해당 파트를 요약하자면 각 단어를 feature vector($ℝ^m$)로 변환하고 joint probability function로 표현하여 단어들의 feature vector와 probability function의 parameter를 동시에 학습함

<br>

- feature vector는 각 단어를 vector 공간 상의 한 점으로 표현할 수 있게 하며 표현된 feature는 어휘(Vocabulary)보다 크기가 훨씬 작음
- 유사한 단어들 간엔 유사한 feature vector를 가지며 probability function는 이러한 feature vector에 대해 smooth한 functoin이기 때문에 feature의 작은 변화는 probability에도 작은 변화를 가져옴 <br>
  -> 하나의 문장에 대해 그 문장만의 probability만 증가하는게 아니라 feature vector 공간 상의 가까운 유사한 문장들의 probability도 함께 증가하게 됨

<br>

### Relation to Previous Work 

- 신경망을 사용하여 고차원의 이산 분포를 모델링하는 아이디어는 이미 입증된 바 있음

- 본 논문에서는 개별 단어를 학습하는 것이 아닌 단어 시퀀스의 확률 분포를 학습하는데 집중하며, 단어 간의 유사성을 표현하기 위해 continuous real-valued vectors를 사용함

<br>

## The Proposed Model: two Architectures 

<br>

- 본 논문에서 목표는 주어진 문맥(이전 단어들)이 주어졌을때 다음 단어 확률 $\hat{P}(w_t|w^{t-1}_1)$을 구하는 것
- 문장을 구성하는 단어 시퀀스의 joint probability를 학습하는 것
- 이를 위한 모델 $f$는 다음과 같음 -> $f(w_t, ..., w_{t-n}) = \hat{P}(w_t|w^{t-1}_1)$
- 모델 $f$의 조건 : 어휘의 단어에 대한 $f$ 값의 합은 1이어야 함, $∑^{|V|}_{i=1} f(i, w_t-1, ..., w_{t-n+1}) = 1$

<br>

- 모델 구성
  - word embedding mapping C:
    - 각 단어 w를 $R^m$ 차원의 실수 벡터로 변환하며 모든 단어는 embedding matrix C의 행을 저장됨 (이때 크기는 $|V| × m$)
    - 단어의 의미적/문법적 유사성을 반영하여 모델이 새로운 단어 조합을 보다 잘 일반화할 수 있도록 함
  - probability function g(or h):
    - C를 통해 변환한 embedding vector 기반 주어진 문맥에서 다음 단어가 나타날 확률 예측하는 함수
    - Direct Architecture: $function g$을 통해 이전 n개의 단어 embedding vector를 입력받아 단어 분포를 출력함
    - Cycling Architecture: $function h$을 통해 문맥에 있는 word vector와 각 후보 word의 vector도 함께 이입력하여 점수 계산함

<br>

- 모델 구조
![image](https://github.com/user-attachments/assets/51b43f37-1150-4b24-ab87-da3aab021a40)
```
- w_t는 t번째 등장하는 단어
- t는 예측할 단어가 등장하는 위치, n은 입력되는 단어 개수

 [Input layer]
- t-(n-1) 부터 t-1번째 단어 벡터 위치를 나타내는 one-hot vector오 matrix C를 내적으로 곱하여 해당 단어의 vetor 값 x_k을 구함
- 구한 x_k를 연결하여 x로 나타냄

[hidden layer]
- tanh 함수를 이용하여 아래의 공식을 통해 score vetor 값을 구함
- y = b + Wx + U*tanh(d+Hx) (b, W, U, d, H는 parameter)

[output layer]
- y 값에 softmax 함수를 적용시켜 정답 one-hot vector와 비교한 뒤 back-propagation를 통해 학습함
```


<br>

## Speeding-up and other Tricks

<br>

- Short list: 다음 단어의 확률을 예측하는데 연산량이 선형적으로 증가하므로 확률이 가장 높은 단어들의 short list를 만들어서 해당 리스트 내에서 상대적인 확률만 계산함
- Table look-up for recognition : 자주 등장하는 입력 문맥에 대해서는 미리 hash table에 저장하여 즉시 참조 가능하도록 함
- SGD




