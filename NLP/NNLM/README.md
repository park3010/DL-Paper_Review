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

#### 차원의 저주

- 이산 확률 변수간의 결합 분포 모델링 시 '차원의 저주'로 인한 학습 문제 어려움이 발생함
  - 단어 수가 100,000개인 자연어에서 10개의 연속된 단어의 결합 분포 모델링 시 가능한 free parameters는 $100,000^10 - 1 = 10^50 - 1$ 개
 
<br>

- 통계적 Language model에서 단어 시퀀스에서 이전 단어들이 주어졌을때 다음 단어가 나타날 조건부 확률로 표현함
  ![image](https://github.com/user-attachments/assets/9428feb7-90da-4f52-b839-cdae58d6231a)
  - $w_t$ 는 $t$번재 단어, 부분 시퀀스는 $w^i_t = (w_i, w_{i+1}, ..., w_{i-1}, w_i)$로 표현
 
<br>

- 









