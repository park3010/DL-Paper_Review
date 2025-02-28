# Efficient Estimation of Word Representations in Vector Space

<br>

## Abstract

<br>

- 본 논문에서는 매우 큰 데이터셋에서 연속적인 단어 벡터(continuous vector representations of words)를 계산하기 위한 2가지 모델 아키텍처를 소개함
- 이러한 표현의 품질은 단어 유사도로 측정하며, 그 결과 훨씬 낮은 계산 비용으로도 정확도가 매우 향샹됨을 확인함
- 또한 단어 벡터가 syntactic & semantic 단어 유사도 측정을 위한 test set에서도 최첨단 성능을 보임

<br>

## Introduction

<br>

- 이전 NLP 시스템에선 단어를 가장 작은 원자 단위로 취급하여 vocabulary 내 인덱스로 표현함 -> 단어간 유사도 개념이 존재하지 않음
- 이러한 방법은 대규모 데이터셋에서 단순한 모델을 학습하는 것이 더 적은 데이터에서 훈련한 복잡한 모델보다 성능이 더 높다는 점에서 장점을 가짐
- but 이러한 단순한 기법은 다양한 task에서 제한됨

<br>

- 본 논문의 목표는 수십억 개의 단어와 수백만 개의 vocabulary를 포함하는 대규모 데이터셋에서 고품질의 단어 벡터를 학습할 수 있는 기법을 소개하는 것임
```
- i.e. 단어 간 유사성은 syntactic 규칙성을 넘어서 단어 벡터에 대한 대수적 연산(+, -, ...)을 통해서도 의미적 의미를 반영할 수 있음
   - $vector("King") - vector("Man") + vector("Woman") ≈ vector("Queen")$

=> 위와 같이 벡터 연산 정확도 향상을 위한 단어 간 선형적 관계(linear regularities among words)를 보존하는 새로운 모델 아키텍처를 개발함
```

<br>

##  Previous Work

<br>

- 단어를 continuous vectors로 표현하는 방법으론 NNLM 연구가 대표적임
  - linear projection layer & hidden layer 를 갖춘 feed forward neural network를 활용하여 word vetor representations과 언어 모델을 같이 학습하는 방식
  - hidden layer를 가진 신경망으로 word vector를 학습한 뒤 해당 벡터를 NNLM에 학습시키는 방식

<br>

##  Model Architectures

<br>

- 본 연구에서는 신경망을 통해 학습된 distributed representations of words에 중점을 두었으며, 이러한 방법은 Latent Semantic Analysis (LSA)보다 linear regularities를 더 잘 보존하며 Latent Dirichlet Allocation (LDA)에선 대규모 데이터에서 높은 계산 비용이 발생한다는 문제를 해결함
- 다른 모델과의 비교를 위해 모델의 computational complexity를 정의함
  - Computational complexity는 모델을 fully train 하는데 필요한 parameter 수로 측정함

```
- Training complexity :

  O = E × T × Q

- E : 훈련 epoch 수
- T : 훈련 데이터셋 내 단어 수
- Q : 모델별 추가적으로 정의되는 변수

- 일반적을 E는 3 ~ 50 사이 값을 가지며, T는 최대 10억 개의 단어까지 포함됨
- 모든 모델은 SGD과 Backpropagation 알고리즘을 학습함
```

<br>

---

<br>

####  Feedforward Neural Net Language Model (NNLM)

![image](https://github.com/user-attachments/assets/db5acdcc-8f55-40b3-954f-7fea6abc6cd0)


```
 [Input layer]
- 1-of-V 인코딩을 사용하여 이전 N개의 단어를 인코딩함
-> 1-of-V 인코딩 : 해당 단어만 1이고 나머지는 0인 벡터로 변환

 [Projection layer]
- hidden layer는 활성화함수로 비선형성을 띄나 Projection layer는 가중치와 행렬의 연산만 이루어지고 활성화 함수는 존재하지 않음
- 공유된 투영 행렬(shared projection matrix)을 사용하여 input layer를 N × D 차원으로 P에 투영함
- 한 번에 N개의 입력만 활성화하므로 계산 비용은 비교적 낮음

 [Hidden layer & Output layer]
- Projection layer의 값은 밀집 벡터이므로 은닉층 계산은 복잡해짐
- 일반적으로 N = 10 으로 설정, Projection layer의 크기는 500 ~ 2000 차원, hidden layer의 크기는 500 ~ 1000 차원 정도로 설정함
- output layer는 vocabulary 내 모든 단어에 대한 확률 분포를 계산해야 하므로 ouput layer의 차원은 V가 됨


- NNLM의 training complexity, Q :

Q = N × D + N × D × H + H × V

N = N개의 단어
D = input layer
H = hidden layer
V = vocabulary 크기

- 계산 비용이 가장 큰 term은 H × V, 계산 비용을 줄이기 위해 vocabulary를 Huffman tree로 표현한 hierarchical softmax 사용
   -> 비용을 log_2(V)로 줄임
```


<br>

####  Recurrent Neural Net Language Model (RNNLM)

![image](https://github.com/user-attachments/assets/55b42a90-22e1-4f94-8e60-894b1ea926c9)

- RNNLM은 context 길이를 사전에 정의해야 하는 NNLM의 한계를 개선함
- RNNLM의 구조는 input, hidden, output layer로 구성되며 projection layer가 없다는 특징이 존재함
- hidden layer에서 자기 자신과 연결하는 시간 지연 연결(time-delayed connections)을 적용해 short-term memory를 갖게 하여 이전 정보를 현재 state에 반영할 수 있도록 함

```
- RNN의 training complexity :

Q = H × H + H × V

- H × H = hidden layer 내부의 순환 연결 연산
- H × V = output layer에서 V에 대한 확률 계산 연산

-> H × V term에 hierarchical softmax를 적용 시 계산 비용을 H × log_2(V)로 줄일 수 있으므로 주요 계산 비용은 H × H에서 나옴
```



<br>

---

<br>

## New Log-linear Models

<br>

- 이번 섹션에서 computational complexity를 최소화하는 두 가지 모델 아키텍처를 소개함
- 위 모델 설명을 통해 대부분의 complexity는 hidden layer에서 발생한다는 것을 확인함, 본 연구에선 simple model을 통해 신경망보다 hidden layer의 표현력이 떨어지지만 더 많은 데이터를 효율적으로 학습할 수 있는 방법을 연구함


<br>

---

<br>

####  Continuous Bag-of-Words Model (CBOW)

![image](https://github.com/user-attachments/assets/ffe7642b-f038-40ec-a095-435a48c3e0d2)

- NNLM과 유사하나 hidden layer는 제거하고 projection layer에서 모든 단어들과 공유함
- 즉, 모든 단어가 동일한 위치로 투영(projection)되어 벡터가 평균화됨
- 이러한 구조는 단어의 순서는 투영 과정에서 영향을 미치지 않기 때문에 Bag-of-Words라고 함(이전에 projection된 단어는 영향을 주지 않음)

- log-linear classifier에 4개의 미래 단어와 4개의 과거 단어를 입력하여 주어진 중심 단어를 분류하는 작업에서 가장 좋은 성능을 보임

```
- CBOW의 training complexity :

Q = N × D + D × log_2(V)

```

<br>

- 기존 Bag-of-Words 모델과 다른 점은 문맥을 continuous distributed representation으로 사용함
- input layer에서 projection layer로 projection할 때 모든 단어에 동일한 weight matrix를 사용함


<br>

####  Continuous Skip-gram Model (CBOW)

![image](https://github.com/user-attachments/assets/f4808a64-532f-4acc-ac5a-92e4ff38dba2)

- CBOW이 문맥 중심으로 과거 단어 n개와 미래 단어 n개를 통해 중심 단어를 예측하는 방식이라면 Skip-gram은 입력한 단어를 바탕으로 입력한 단어의 주변 범위의 단어들을 예측하는 방식임
- Skip-gram은 예측 범위를 늘리면 벡터의 품질은 증가하나 computational complexity는 증가함
  - 현재 단어와 멀리 떨어진 단어일수록 연관성이 낮으므로 멀리 위치한 단어를 샘플링할 확률을 낮춰 학습 예제에서 멀리 위치한 단어들의 비중을 낮춤

```
- Skip-gram의 training complexity :

Q = C × (D + D × log_2(V))

- C = 단어 간 최대 거리
```

<br>

---

<br>

## Result

<br>

- 기존 연구는 단어 벡터 간 비교를 위해 유사한 단어 쌍끼리 테이블로 나열한 표를 사용함
- 이전 연구에서는 단어들 간 다양한 유사성이 존재함을 발견함
  ```
  - "big" → "bigger"는 "small" → "smaller" 와 동일한 관계를 가짐
  - "big" → "biggest"는 "small" → "smallest" 와 같은 관계도 가짐
  ```

- 이러한 관계는 단순한 벡터 연산을 통해서 해결 가능
  ```
  X = vector("biggest")−vector("big")+vector("small")
  ```
  -> 그리고 벡터 공간 상에서 X와 코사인 유사도가 가장 큰 단어를 찾아서 정답으로 찾음


