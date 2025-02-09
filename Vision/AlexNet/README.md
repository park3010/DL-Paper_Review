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
    
- 최근에 와서 수백 만 개의 라벨링된 데이터셋 수집 가능해짐 -> 수십만 개의 fully-segmented image로 구성된 ImageNet과 22,000개의 카테고리와 1,500만 개 이상의 라벨을 가지는 고해상도 이미지로 구성된 ImageNet이 포함됨<br>
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

- ImageNet은 다양한 해상도의 이미지를 포함하지만 신경망에 입력 사이즈를 맞추기 위해 256×256 크기의 이미지로 다운 샘플링 작업함<br>
  ![image](https://github.com/user-attachments/assets/e83dbfd1-9d81-4901-a0a6-f255bfdeb759)<br>
  -> 본 연구에선 train set의 평균 빼는 것말곤 별도의 전처리 작업은 진행하지 않음

<br>

## ReLU Nonlinearity

<br>

- 입력 x에 대한 출력 f는 `f(x) = tanh(x)` or `$$f(x) = (1 + e^-x)^-x$$` 임
  - 경사 하강법을 이용한 훈련시간면에서 위 방법은 `f(x) = max(0,x)` 보다 느림 => ReLU Nonlinearity
  - Deep convolutional neural networks(DCNNs)에서 `ReLU`는 동등한 `tahn`보다 학습 속도가 빠름 -> 빠른 학습은 큰 모델과 많은 데이터셋에 효과적임<br>
  ![image](https://github.com/user-attachments/assets/c90eccf5-c79e-4e9e-9b5c-2c9a86bd196d)<br>
```
4개의 ReLU를 이용한 4개의 CNN계층만으로도 CIFAR-10 데이터에 대해 6번 에포크만에 25%의 에러율을 달성함
```

<br>

## Training on Multiple GPUs

<br>

- Single GPU의 메모리는 3GB 밖에 되지 않으므로 본 연구에서는 2개의 GPU를 사용함
- GPU parallelization(병렬화)는 kernel을 반으로 나눠서 각 GPU에 할당하며 GPU 간의 communication은 특정 layer에서만 발생하도록 함<br>
  -> layer 2,4,5에서는 바로 이전의 동일한 GPU에서 연사된 결과만 가져온 반면, layer 3에서는 GPU communication을 통해 이전 layer에서 연산된 2개의 GPU 결과를 모두 받아옴

- 성능 : 하나의 GPU를 사용하는 conv 계층의 kernel 수가 절반인 네트워크보다 top-1, top-5에서 각각 error 1.7%와 1.2%를 감소시킴, 훈련 시간도 더 빠름

<br>

## Local Response Normalization

<br>

- ReLU는 gradient saturating(포화)를 방지하기 위해 입력 정규화를 할 필요 없으나 아래의 `local normalization`을 통해 normalization에 도움됨
  -> 입력 정규화를 진행하는 이유 : 입력값 범위가 다를 경우 Gradient Descent Algorithm 적용이 까다로워짐(최적화하기 힘듬)

<br>

---
<br>

<div aglign="centor">
  Local Response Noramlization(LRN) 이해를 위한 추가 설명
</div>

<br>

- Local Response Noramlization layer는 측면 억제(lateral inhibition)를 구현한 것임

  - 측면 억제(lateral inhibition): 한 영역에 있는 신경 세포가 상호 간 연결되어 있을 때 한 그 자신의 축색이나 자신과 이웃 신경세포를 매개하는 중간신경세포(interneuron)를 통해 이웃에 있는 신경 세포를 억제하려는 경향<br>
    ![image](https://github.com/user-attachments/assets/2cef4689-953f-4668-bf85-a8383015e9e6)<br>
    ```
    흰색 선에 집중하지 않고 그림을 보게 될 경우 회색의 점이 보이는데 이러한 현상은 측면 억제(lateral inhibition)에 의해 발생함
    -> 흰색으로 둘러싸인 측면에서 억제를 발생시켜 횐색이 더 반감되어 보인 것
    ```
<br>

---

<br>

- ReLU 함수는 입력값을 양수가 들어올 시 입력값 그대로 출력하기 때문에 합성곱이나 pooling 연산 이후 ReLU 함수 적용 시 매우 높은 하나의 픽셀 값이 주변의 픽셀에 영향을 미칠 수 있으므로 feature map에서 인접한 위치에 있는 픽셀끼리 정규화 해줌<br>
![image](https://github.com/user-attachments/assets/6eab8328-192f-4124-93fa-61bdaf6a5379)

<br>

## Overlapping Pooling

<br>

- 일반적으로 풀링 시 뉴런이 중복되지 않도록 `stride = pooling size` 풀링을 진행하나 본 연구에서는 `stride = 2 < pooling size = 3` 으로 설정해 풀링되는 뉴런이 중복되도록 진행하여 error 감소 및 과적합 감소 효과를 얻음<br>
![image](https://github.com/user-attachments/assets/e53e6b2e-b81c-4df5-98fa-6bcaee8ff470)

<br>

## Overall Architecture

<br>

- 모델 구성: 5개의 Convolutional Layer와 3개의 Fully Connected(FC) Layer로 구성되며 FC layer의 마지막은 softmax 함수를 사용하여 1000개의 output을 출력함
- 2, 4, 5번째 conv layer는 이전 레이어의 같은 GPU의 값만 입력 값을 받으나 3번째 conv layer에서는 모든 값을 입력받음<br>
![Screenshot_20250209_132438_Samsung Notes.jpg](https://github.com/user-attachments/assets/41a62b72-e490-4406-a27b-967112fe442c)<br>

  [Conv1]
    - input : 224x224x3
    - filter : 11x11x3. 96개. 4-stride
    - activation : ReLU + LRN + MaxPooling
  [Conv2]
    - filter : 5x5x48. 256개
    - activation : ReLU + LRN + MaxPooling
  [Conv3]
    - filter : 3x3x256. 384개
    - activation : ReLU
    -> 유일하게 이전 layer에서 모든 kernel map들과 연결굄
  [Conv4]
    - filter : 3x3x192. 384개
    - activation : ReLU
  [Conv5]
    - filter : 3x3x192. 256개
    - activation : ReLU
  [FC6]
    - Neurons : 4096
    - activation: ReLU
  [FC7]
    - Neurons : 4096
    - activation : ReLU
  [FC8]
    - Neurons : 1000
    - activation : Softmax
