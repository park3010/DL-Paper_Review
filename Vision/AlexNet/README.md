# ImageNet Classification with Deep Convolutional Neural Networks

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

  - [Conv1]
    - input : 224x224x3
    - filter : 11x11x3. 96개. 4-stride
    - activation : ReLU + LRN + MaxPooling<br>
  - [Conv2]
    - filter : 5x5x48. 256개
    - activation : ReLU + LRN + MaxPooling<br>
  - [Conv3] -> 유일하게 이전 layer에서 모든 kernel map들과 연결됨
    - filter : 3x3x256. 384개
    - activation : ReLU<br>
  - [Conv4]
    - filter : 3x3x192. 384개
    - activation : ReLU<br>
  - [Conv5]
    - filter : 3x3x192. 256개
    - activation : ReLU<br>
  - [FC6]
    - Neurons : 4096
    - activation: ReLU<br>
  - [FC7]
    - Neurons : 4096
    - activation : ReLU<br>
  - [FC8]
    - Neurons : 1000
    - activation : Softmax<br>

<br>

## Reducing Overfitting

<br>

- 본 논문에서 60만 개의 parameter가 있으므로 Overfitting 피하기 위한 2가지 기법인 `Data Augmentation`과 `Dropout`를 소개함

<br>

### Data Augmentation

<br>

- 가장 쉬운 Overfitting 감소 방법은 데이터셋 늘리는 것 -> 연산량이 작으므로 GPU로 학습하는 동안 CPU에서 계산하여 Augmentation함

  - 두 가지 주요 방법
    1. `이동`과 `좌우 반전`
      - 256x256 이미지 내에서 224x224 크기의 patch를 무작위로 자른 뒤 훈련 시킴 -> 기존 훈련 세트에서 2048배 증가
      - 테스트 진행할 땐 224 x 224크기의 patch 5개(좌상단, 우상단, 우하단, 좌하단 꼭지점에 붙인 4개의 patch와 가운데 patch 1개)와 이를 좌우 반전시킨 10개의 patch를 사용함

    2. RGB 채널 강도 조절
       - RGB 픽셀에 PCA를 적용하여 N(0,1) 정규 분포에서 추출한 랜덤값을 기존 픽셀에 더하여 색상 조정함

<br>

### Dropout

<br>

- 기존에는 여러 모델의 예측값을 합치는 앙상블 기법을 많이 사용했으나 이 방법은 매우 큰 신경망에서는 비용이 높게 나옴
- 은닉층의 뉴런을 0.5확률로 0으로 설정하는 `Dropout`은 훈련 비용은 2배지만 매우 효과적인 모델 결합 방법임
  - 'Dropped out'된 뉴런은 forward-pass와 back-propagation에서 영향을 미치지 않게 됨
  - 본 논문에서는 FC layer 2개에 Dropout을 적용함


<br>

## Reducing Overfitting

<br>

- batch size = 128
- stochastic gradient descent(SGD)
  - momentum = 0.9
  - weight decay = 0.0005 -> 정규화 뿐만 아니라 학습 오차도 감소시켜줌
- weight initialized
  - 표준편차 = 0 의 zero-mean Gaussian distribution(zero-mean 가우시안 분포)를 각 layer에 적용함
  - 2, 4, 5번째 conv layer와 FC layer의 bias는 1, 나머지는 0으로 초기화함 -> 학습 가속화 적용
- Learning rate
  - 모든 layer에서 동일한 학습률을 적용하나 훈련 수행 중 매뉴얼하게 조정함
  - 0.001로 설정하였으며 현재 학습률로 validation error가 개선되지 않으면 10으로 나눔 -> 총 3번 감소함

<br>

## Result

<br>

![image](https://github.com/user-attachments/assets/cc9cc286-02d5-4221-96be-3721977176d8)

```
ILSVRC-2010 대회의 테스트 데이터셋에 대한 결과
  -> Sparse coding : 서로 다른 특징을 학습한 6개의 sparse-coding model에서 나온 예측한 값에 대한 평균을 구함
  -> SIFT + FVs : Fisher Vectors(FVs)로부터 훈련된 두 개의 분류기의 예측값을 평균함
```
<br>

![image](https://github.com/user-attachments/assets/fbe18ab9-f3ef-4307-bb0a-4404f9523afe)

```
Validation set과 Test set을 예측했을 때 정확도 차이가 0.1% 이하로 나와서 두 지표를 같이 사용하여 비교함
  -> 1CNN은 CNN 모델 한 개 사용한 것, 5CNN은 5개의 CNN 모델 예측값을 평균낸 것
  -> 1CNN*과 5CNN* 모델은 mageNet2011의 전체 데이터에 대해 사전학습된 모델임
=> 전반적으로 CNN 모델 개수가 많을수록 오차율이 낮게 나옴
```

<br>

### Qualitative Evaluations

<br>

![image](https://github.com/user-attachments/assets/d29c27d1-05f4-42cb-bd7c-2521637301a4)

```
Figure 3: 224X224X3 의 이미지 입력을 받아 11X11X3 크기의 96개의 커널을 가진 첫번째 합성공계층을 통과한 것이다. 위 3줄은 첫 번째 GPU에서, 아래 3줄은 두 번째 GPU에서 작동한 것
```

- 첫 번째 GPU에선 주로 색상이 거의 인코딩되지 않았지만 두 번째 GPU에선 색상 인코딩이 진행됨

<br>

![image](https://github.com/user-attachments/assets/a60a1f9f-6a8e-463a-aab9-f0566334bbe2)

```
Figure 4: (왼쪽) 8개의 ILSVRC-2010 테스트 이미지에 대해 AlexNet이 예측한 가장 유력한 5가지 Label, 정답은 각 이미지의 아래에 표시되어 있고 그에 대한 확률은 빨간색으로 표시함 
(오른쪽) 첫번째 열에 위치한 ILSVRC-2010의 테스트 이미지에 대해 나머지 각 열의 이미지 5개는 마지막 은닉층에서 특성벡터에 대한 유클리드 최소거리를 이용하여 얻은 훈련 이미지임
```

- figure 4를 통해 AlexNet은 중앙에서 벗어난 객체 또한 감지할 수 있으며 top-5에 속하는 예측값이 실제값과 유사한 것을 통해 일리있는 예측을 선보임

<br>

### Discussion

<br>

- 본 논문에선 순수 지도학습 만으로도 CNN은 높은 성능을 보였으며, convolutional layer를 하나씩 제거할 때마다 top-1 Accuracy 가 2%씩 감소한 것을 이유로 CNN layer가 깊을수록 효과적임을 강조함
