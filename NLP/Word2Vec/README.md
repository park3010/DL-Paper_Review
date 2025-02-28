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

####  Feedforward Neural Net Language Model (NNLM)

- ㅇㄹ
