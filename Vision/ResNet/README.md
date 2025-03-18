# Deep Residual Learning for Image Recognition

<br>

## Abstract

<br>

- 딥러닝 모델에서 네트워크가 깊어질수록 학습이 어려워지므로 이를 완화하고자 본 논문에선 잔차 학습(residual learning) 프레임워크를 제안함
- 잔차 네트워크(residual networks, ResNet)가 더 최적화가 쉬우며, ImageNet Dataset에서 최대 152개의 layer를 가진 네트워크를 실험했을 때 VGGNet(19~22 layer)보다 8배 더 깊지만 계산 복잡도는 더 낮은 걸 확인함

<br>

## Introduction

<br>

- 딥러닝 네트워크는 저/중/고수준의 feature를 layer를 통해 통합함 -> 이는 layer가 깊어질수록 더 풍부한 표현 학습 가능하다는 의미
- but layer가 깊어질수록 gradient vanishing 또는 exploding gradients 문제가 발생하게 되며 이로 인해 훈련이 어려워짐
  ![image](https://github.com/user-attachments/assets/31f6d2d9-fb71-46fa-98a0-b7dcc054d5a9)
- normalized initialization 등을 통해 이러한 문제 완화 가능하나 여전히 layer가 깊어질수록 성능이 급격히 감소하는 문제 발생함

<br>

- 본 논문에서는 Residual Learning Framework를 제안함 <br>
  ![image](https://github.com/user-attachments/assets/2d04bc71-c895-41c1-9251-72d032c0347c)
  ```
   [residual learning]
  입력값 x를 받아 타겟값 y로 mapping하는 H(x)를 학습하고자 할 때

  - 기존 layer : 입력 x를 받아 layer를 거쳐 H(x)를 출력함
  - ResNet layer : layer가 H(x)를 직접 학습하는 것이 아닌 출력과 입력의 잔차인 F(x) := H(x) - x 를 학습하도록 설계함
    -> 즉, F(x) := H(x) - x를 원래 학습하고자 했던 함수 H(x) = F(x) + x로 변환 가능

  - F(x) = H(x) - x를 학습하는 만큼 이를 최소화시켜야 하며, 이는 출력과 입력의 차를 줄여야 함을 의미함
  - F(x) = 0이 최적의 해가 되며 0 = H(x) - x, H(x) = x로 mapping하는 것이 학습 목표가 됨

  - 이러한 잔차 학습(residual learning)은 shortcut connection 형태로 구현됨
  ```

<br>

## Related Work

<br>

### Residual Representations

- vector quantization 에서 original vector를 encoding 하는 것 보다 residual vector를 encoding 하는 것이 훨씬 효과적임
- vector quantization : feature vector x를 class vector y로 mapping 하는 것을 의미함

<br>

### Shortcut Connections

- 본 논문에서의 shortcut connection은 추가적인 parameter를 필요로 하지 않으며 0으로 수려하지 않으므로 절대 닫힐 일 없이 항상 정보를 그대로 전달함
- 이를 통해 residual function으로는 지속적으로 학습이 가능함


<br>

## Deep Residual Learning

<br>

- 앞서 언급한 바와 같이 residual function F(x)는 H(x) - x 로 근사할 수 있으며 이를 통해 H(x) = F(x) + x 로 변환 가능함
- shortcut connection 또한 추가적인 parameter와 computational complexity를 필요로 하지 않음
- residual learning 을 layer에 적용하기 위해선 입력 x와 F는 같은 차원을 가지고 있어야 함 -> 만약 차원이 다를 경우 linear projection $W_s$ 적용하여 차원을 맞출 수 있음
  ```
  y = F(x, {W_i}) + W_s * x
  ```

<br>

## Network Architectures

<br>

![image](https://github.com/user-attachments/assets/67dfd2d8-a4e3-4559-ad25-61dac39bcacc)

<br>

### Plain Network

- plain network는 VGGNet에 영감을 받았으며 아래 두 가지 규칙에 기반하여 설계함
  ```
  - output feature map의 size 와 동일한 수의 conv filter를 사용함
  - output feature map의 size 가 반으로 줄어들면 time complexity를 동일하게 유지하기 위해 filter 수를 두 배로 증가시킴
  ```
- 이 plain network의 구조는 34개의 weight layer로 구성되어 있으며 마지막 부분은 global average pooling layer와 1000개의 뉴런을 가진 fully-connected layer + softmax 활성화 함수로 이루어짐


<br>

### Residual Network

- residual network는 shortcut connection을 추가한 구조이며 입력과 출력 차원이 동일한 경우 identity shortcut을 적용할 수 있음
- 입력과 출력이 다를 경우 아래의 두 가지 방법을 적용할 수 있음
  ```
  - zero padding 기법을 사용하여 추가된 차원의 수를 일치시킨 뒤 identity shortcut을 적용함
  - linear projection W_s 를 곱하여 projection shortcut을 사용하여 차원 일치시킴 -> 1x1 convolution 으로 구현
  ```

<br>

## Implementation

<br>

- 이미지 크기 조정
  - 이미지의 짧은 변 크기가 256 ~ 480 픽셀 범위 내에서 Random하게 Sampling 하여 Resize 함
  - Horizontal flip에서 Random Sample 후 per-pixel mean을 빼줌
  - Standard color augmentation 기법 적용
- 신경망 구조
  - conv layer와 ReLU 함수 사이에 Batch Normalization 적용
  - He initialization 방법으로 가중치 초기화
- 학습 과정
  - Opimizer : SGD
  - mini-batch size = 256
  - Learning rate = 초기에는 0.1로 시작함(error가 일정 수준 이상 감소하지 않으면 10배 감소)
  - weight decay = 0.0001, momentum = 0.9
  - Dropout 사용 안함
  - 최대 반복 횟수 : $60 × 10^4$
- 테스트
  - 10-crop testing 기법 적용
  - 이미지의 짧은 변 크기를 {224, 256, 384, 480, 640} 중 하나로 resize한 뒤 평균을 계산함
 

<br>

## Experiments

<br>

### ImageNet Classification

![image](https://github.com/user-attachments/assets/6742a090-89fe-435c-a243-ea5ff629271e)
![image](https://github.com/user-attachments/assets/19ca56c8-eeac-4f3c-81b8-444b19397647)

<br>

### Plain Networks

- 34-layer의 training error > 18-layer의 training error  -> 성능 저하 문제(degradation problem)가 발생했기 때문
- 본 실험의 plain network에 Batch Normalization를 적용하여 forward propagation이 0이 되지 않도록 보장했으며, back-propagation 과정에서도 gradient가 소실되진 않음
- 즉, layer가 깊어질수록 지수적으로 낮은 수렴 속도(exponentially low convergence rate)로 인해 최적화가 어려워졌음을 의미함

<br>

### Residual Networks

- 본 실험에서 모든 shortcuts에 identity mapping 을 적용하고 차원 맞추기 위해 zero-padding 적용함
- 34-layer에서도 성능 저하 문제(degradation problem) 피할 수 있었으며 빠른 수렴 속도를 가짐 


<br>

### Identity vs. Projection Shortcuts

![image](https://github.com/user-attachments/assets/0f2516a8-7ade-4a71-9f5c-0f9c154ec558)

- (A-C) 옵션을 통해 projection shortcut 성능 비교함
- (A) 차원 증가 시 zero-padding 적용하여 shortcut connection 유지함
- (B) 차원 증가 시 projection shortcut을 적용하며 나머지 shorcut은 identity shortcut을 적용함
- (C) 모든 shortcut에 projection shortcut 적용함
- (A-C) 옵션 간 성능을 비교했을 때 (C) 옵션의 성능이 가장 나았으나 유의미한 차이는 없음
- (C) 옵션 적용 시 computation complexity를 증가시키므로 이후 실험에선 (B) 옵션을 적용함

<br>

### Deeper Bottleneck Architectures

![image](https://github.com/user-attachments/assets/29878d0a-56eb-4edd-9907-bd0fe94d9b51)


- 본 논문에서는 그림 5의 오른쪽처럼 bottleneck 구조로 수정하여 성능을 테스트함
- 기존 residual function F(x) 구조가 2개의 layer로 구성되어 있다면 bottleneck design은 3개의 layer로 구성됨
  - 1x1 conv -> 3x3 conv -> 1x1 conv
  - 1x1 conv layer로 차원 축소 및 증가시키고 3x3 conv layer로 bottleneck 역할 수행함
 
- bottleneck 구조에서 identity shortcut을 적용 시 모델 크기 및 연산량을 2배 증가시켜 더 효율적인 모델 설계를 가능하게 해줌


