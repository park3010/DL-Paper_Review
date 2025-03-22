# Rethinking the Inception Architecture for Computer Vision

<br>

## Abstract

<br>

- 대부분의 task에선 라벨링된 훈련 데이터가 충분하다고 가정했을때 모델 크기와 깊이가 깊어질수록 성능은 향상하나 그만큼 연산 비용이 커지게 됨
- computational efficiency과 low parameter count가 중요한 환경에선 모델 크기와 깊이는 단점이 됨
- 본 연구에선 자원을 효율적으로 활용할 수 있도록 convolutions을 분해(factorization)하고 공격적인 정규화(aggressive regularization)를 적용하여 네트워크 확장하는 법을 소개함

<br>

## Introduction

<br>

- Inception 기반 GoogLeNet의 파라미터 수는 500만 개로 6천만 개의 AlexNet이나 그보다 3배 이상 많은 VGGNet보다 적은 수를 가짐
- 효율성 측면에서 Inception은 다른 network보다 좋으나 구조적 복잡성 때문에 네트워크 변경에 어려움이 존재함
- 다음 섹션에서 convolution network의 효율적 확장을 위한 general principle과 최적화 아이디어를 소개함

<br>

## General Design Principles

<br>

1. 네트워크 초반부에서의 표현 병목(representational bottleneck) 피하기
   - Feed-forwoard network는 input layer -> classifier or regressor 로 이어지는 비순환 그래프로 표현 가능하며 명확한 방향의 정보 흐름으로 정의됨
   - 보통 표현의 크기(차원의 수)는 input에서 output layer로 갈수록 점진적으로 감소해야 하나 이때 극단적으로 압축된 병목을 피해야 함

 <br>

2. Higher dimensional representations은 locally하게 처리하기 쉬움
   - conv layer에서 tile마다 activation 수를 늘리면 더 분할된(disentangled) feature를 학습할 수 있음

 <br>

3. 공간적 집계(spatial aggregation)는 저차원 임베딩에서도 큰 손실없이 수행 가능
   - 인접 유닛 간의 강한 상관관계를 통해 차원 축소 시 정보 손실이 적기 때문에 더 넓은 범위의 (e.g 3x3) conv 수행 전 입력 표현의 차원을 줄이더라도 큰 부작용은 없을 것이라 가정함

 <br>

4. network width와 depth의 균형 맞추기
   - 각 stage별 filter 수(폭)와 전체 깊이 간의 균형을 통해 최적의 네트워크 성능 달상 가능함


<br>

## Factorizing Convolutions with Large Filter Sizes

<br>

- GoogLeNet의 경우 1x1 conv layer로 차원 축소 후 3x3 conv layer 여러 개를 적용하여 local representations은 유지하면서 파라미터 수를 줄일 수 있었음


<br>

### Factorization into smaller convolutions

<br>

![image](https://github.com/user-attachments/assets/dedddd4c-c8b2-4f0d-857a-7ff8f232e115)

- 5x5 convolution, 7x7 convolution의 경우 3x3 convolution으로 분해하면 같은 input size와 output depth를 유지하면서 더 작은 연산량과 파라미터를 구현할 수 있음
- 위 그림처럼은 5x5 convolution는 3x3 convolution 두 개(첫 번째는 conv, 두 번째는 fc)로 대체할 수 있음  <br>
  <img src="https://github.com/user-attachments/assets/6d075166-b889-41e3-ab43-c14721211324" width="300" height="300">
  <img src="https://github.com/user-attachments/assets/6a2e13b5-07d7-4ce3-a5b8-030138ac4955" width="300" height="300">

<br>

- 5x5 convolution을 두 개의 3x3 convolution으로 분해했을 때 첫 번째 3x3 conv에는 linaer activation, 두 번째 3x3 conv에는 ReLU activation을 사용하는 것보다 둘 다 ReLU activation을 사용하는 것이 더 성능이 높게 나옴  <br>
  <img src="https://github.com/user-attachments/assets/a972c0f8-5105-4d5f-909c-d267728aadef" width="350" height="300">

<br>

### Spatial Factorization into Asymmetric Convolutions

<br>

<img src="https://github.com/user-attachments/assets/610a3df8-57ec-4eb6-a6d1-8b28f2acf01f" width="300" height="350">

- 3x3 convolution을 더 작은 2x2 convolution으로 분해할 수 있는가?
- 실제 실험했을때 2x2 convolution으로 분해했을 때보다 nx1 비대칭 convolution으로 분해하는게 더 효과적임
- 3x3 convolution의 경우 3x1과 1x3 convolution으로 연속적으로 분해할 수 있으며 이럴 경우 33%의 연산량 절감 효과가 있음, 이는 2x2 conv을 통해 절감한 11%보다 더 큼

<br>

- nxn convolution을 nx1과 1xn convolution으로 대체했을 때 faeture map 사이즈가 12~20 사이일 때 효과가 가장 큼 <br>
  <img src="https://github.com/user-attachments/assets/de261a3c-9714-4b2e-a640-dd4fd612b10d" width="300" height="350">


<br>

## Utility of Auxiliary Classifiers

<br>

![image](https://github.com/user-attachments/assets/70179934-6a7d-47fc-98d0-38fdc66d359a)

- GoogLeNet에서 Auxiliary Classifiers을 활용하면 신경망 수렴을 개선하는데 도움을 준다고 소개하나 실제 실험 결과 큰 차이는 없는 걸 알게 됨
- Auxiliary Classifiers에 batch normalization이나 dropout을 적용했을 때 main classifier의 성능이 향상됨
- 이를 통해 저수준 feature 학습에 도움을 주는 것이 아닌 정규화 효과를 주는 것으로 추측됨


<br>

## Efficient Grid Size Reduction

<br>

- 기존 CNN 신경망은 feature map의 grid 크기를 줄이기 위해 pooling 연산을 하며, 이때 표현 병목(representational bottleneck)을 피하기 위해 max pooling 또는 avg pooling을 적용하기 전 채널 수를 증가시킴 <br>

  ![image](https://github.com/user-attachments/assets/953dbe54-4c94-4cc2-9d7a-ce47bfc26e18)

  ```
  d x d 크기를 가진 k개의 feature map을 (d/2) x (d/2) 크기를 가진 2k개의 feature map으로 만들고자 할 때

  [오른쪽 그림]
  1. stride 1의 convolution 적용하여 필터 수를 2k개를 늘린 후
  2. pooling layer를 지나서 크기를 절반으로 줄임
  => 연산량을 계산 시 2d^2 k^2 가 됨

  [왼쪽 그림]
  위 방법의 연산량을 줄이기 위해 convolution과 pooling 순서를 적용한 뒤 그 결과에 convolution을 적용함
  => 연산량 계산 시 2(d/2)^2 k^2 가 됨
  이러한 방법은 연산량은 감소시켜 주지만 신경망의 표현력도 감소시킴 
  ```

<br>

- 본 논문에서는 표현력(representation)을 감소시키지 않고 연산량을 감소시키는 방법으로 두 개의 병렬적인 stride 2 구조의 블록을 사용함
  - P block : avg pooling 또는 max pooling layer
  - C block : stride 2를 갖는 convolution layer <br>
    ![image](https://github.com/user-attachments/assets/a08db096-930f-40ac-b28c-e63dfd5190f1)
