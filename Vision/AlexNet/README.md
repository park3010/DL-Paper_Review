# ImageNet Classification with Deep Convolutional Neural Networks

<br>

```plaintext
한 줄 요약 : 
```

<br>

## Abstract

<br>

- 본 논문에서 제안하는 AlexNet을 사용하여 ImageNet LSVRC-2010 대회에서 12만 개의 고해상도 이미지를 1000개의 class를 분류함
  - top-1, top-5에서 각각 error 37.5%와 17.0%를 달성함
  - 이 수치는 기존 최첨단 모델보다 좋은 수치임
- neural network 구성
  - 6천만 개의 parameter와 65만 개의 뉴런으로 구성됨
  - max-pooling layer를 동반한 5개의 convolution layer와 마지막에 1000개의 class 구분하는 softmax 적용한 fully-connected layer 3개로 구성
- 학습 속도 향상을 위해 제안한 방법 -> 효율적인 convolution 연산을 위한 GPU 방법과 non-saturating neurons을 사용함
- fully-connnected layer에서의 overfitting 감소 -> "dropout" 정규화 기법 사용함
- ILSVRC-2012 대회에서 변형된 모델을 제출하여 오차율 26.2%를 달성한 2위 모델보다 뛰어난 15.3%를 달성하여 우승함

<br>

## Introduction

<br>

- 수만 개 정도의 작은 규모를 가지는 라벨링된 데이터셋으론 간단한 recognition(인식) 작업은 가능함 but 실제 환경의 데이터는 매우 복잡하므로 더 큰 데이터셋을 필요로 함
  - Example) Single recognition task, MNIST digit-recognition(숫자 인식) 작업의 error는 0.3% 미만으로 사람이 하는 것에 근접함
    
- 최근에 와서 수백 만 개의 라벨링된 데이터셋 수집 가능해짐 -> 수십만 개의 fully-segmented image로 구성된 ImageNet과 22,000개의 카테고리와 1,500만 개 이상의 라벨을 가지는 고해상도 이미지로 구성된 ImageNet이 포함됨
    ```
    fully-segmented image : 이미지의 유사한 영역 또는 부분(segment)를 동일 클레스 레이블로 그룹화한 이미지
    ```
    ![image](https://github.com/user-attachments/assets/907ccbd6-5d80-4c4d-9bd9-8909949bccee)
<br>

- 수천 개의 object가 있는 수백 만 개의 이미지 학습 위해선 학습 용량 큰 모델 필요함
- but, object recognition 작업의 복잡성은 ImageNet 같은 큰 데이터셋으로도 해결 하기 어려움
  -> 본 논문의 모델은 우리가 가지고 있지 않은 데이터를 보완하기 위해서 많은 사전 지식을 갖춰야 함
<br>

- Convolution neural network(CNN)는 깊이(depth)와 폭(breadth)를 다양하게 조절하고 이미지 특성에 대해 대부분 올바르게 예측 가능함
- 비슷한 layer 크기를 가지는 표준 FeedForward neural networks와 CNNs를 비교했을 때
- CNN -> 더 적은 수의 connections와 parameter 수로 훈련 가능하고 성능 또한 우수함
<br>
  
- CNN의 특성과 상대적으로 효율적인 구조에도 불구하고 여전히 매우 큰 고해상도 이미지에 대한 계산 비용이 높음
- 이에 대해 GPU는 2D convolution 연산에 최적화되어 있어 CNN 훈련 가능하고 ImageNet은 심각한 overfitting 없이 모델이 훈련할 수 있도록 충분히 라벨링되어 있음
<br>

```
- 본 논문 성과:
  - 매우 큰 CNN 모델을 학습하여 ILSVRC-2010 및 ILSVRC-2012 대회에서 가장 우수한 성적 달성
  - 2D convolution 연산에 최적화된 GPU를 구현 및 공개함
  - 성능 개선 및 학습 시간 단축시키는 새로운 방법 포함
  - 네트워크 크기로 인해 발생할 수 있는 overfitting 문제 방지하기 위한 기법 소개
  - 본 논문의 network는 5개의 convolution layer와 3개의 fully-connected layer로 구성된 구조가 매우 중요함
    -> 전체 paramter에서 1%도 차지하지 않는 단 하나의 convolution layer 제거했는데도 성능이 떨어짐
```
<br>

## The Dataset

<br>

- ImageNet은 다양한 해상도의 이미지를 포함하지만 신경망에 입력 사이즈를 맞추기 위해 256×256 크기의 이미지로 다운 샘플링 작업함
  ![image](https://github.com/user-attachments/assets/e83dbfd1-9d81-4901-a0a6-f255bfdeb759)
  -> 본 연구에선 train set의 평균 빼는 것말곤 별도의 전처리 작업은 진행하지 않음

<br>

## The Architecture

<br>

본 논문의 신경망은 아래와 같음
![image](https://github.com/user-attachments/assets/41f28cdb-7537-47f7-9d7f-c9d62e2cc63a)
```
  network 아키텍처의 특징
1. ReLU Nonlinearity
2. Training on Multiple GPUs
3. Local Response Normalization
```

<br>

### ReLU Nonlinearity

<br>

