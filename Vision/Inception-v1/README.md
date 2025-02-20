# Going deeper with convolutions

<br>

## Abstract

<br>

- "Inception" 라는 이름의 Deep Convolutional Neural Network를 제안함
- 신중하게 설계된 network를 통해 network의 depth와 width를 증가시키면서 계산 비용을 일정하게 유지시켜 **컴퓨터 내 자원의 활용을 최적화**시킴
- 품질 최적화를 위해 아키텍처는 Hebbian principle와 multi-scale processing에 대한 직관 기반으로 모델 설계함
- Inception 기반 22개의 layer로 구성된 "GoogLeNet"을 구성하여 ILSVRC14 대회의 Image Recgmition 및 Object Detection 분야에서 우수한 성적 달성함

<br>

## Introduction

<br>

- 본 논문에서 ILSVRC 2014 대회에서 제출한 GoogLeNet은 두 해 전 우승한 f Krizhevsky et al의 아키텍처보다 12배 적은 paramters로 더 높은 정확도를 보임
- Object Detection 분야에서의 성능 향상은 deep architectures와 기존 컴퓨터 비전의 시너지 덕분
  - 대표적인 예제로 Girshick et al의 R-CNN이 있음

<br>
  
- 효율성을 고려하여 모델이 추론(Inference) 시 약 15억 번의 곱셈-덧셈 연산 내에서 계산 비용 유지하도록 설계함

<br>

- 컴퓨터 비전을 위한 효율적인 심층 신경망 : Inception
  - 본 논문에서 본 "Deep"의 의미
    - “Inception module" 이라는 새로운 organization(조직화) 방식 도입한다는 점에서 깊음
    - 보다 직접적으로 네트워크 깊이를 증가시킴
  - Lin et al의 연구를 확장한 형태이며, Arora et al의 이론적 연구에서 영감을 받음

<br>

## Related Work

<br>

- CNN은 LeNet-5의 연속적인 Convolution laye layer와 한 개 이상의 Fully-connected layer를 갖는 표준 구조를 따름
- 해당 디자인 기반 변형 모델이 Image Classification 연구에서 널리 사용되며 대규모 데이터셋 처리에선 네트워크의 layer 수 증가, layer의 크기 증가, dropout 기법을 활용한 overfitting 방지 등을 포함함

<br>

- Serre et al은 영장류 시각 피질(Primate Visual Cortex)의 신경 과학 모델에 영감을 받아 다양한 크기의 고정된 Gabor Filter를 활용한 multiple scale 문제 처리 방식을 제안함
  ```
  Inception 모델과 유사함 but
  - Serre et al은 2-layer 모델 사용
  - Inception 모델은 모든 필터 학습, Inception layer이 여러 번 반복됨
    -> GoogLeNet은 22-layer로 구성
  ```

<br>

- Lin et al은 신경망의 표현력 향상을 위해 Network-in-Network 접근 방식 제안함
  ```
  Network-in-Network
  - conv layer에 적용 시 1×1 conv layer와 ReU 함수가 적용된 형태와 유사
  - CNN 파이프라인도 쉽게 통합 가능
  ```
- 본 논문에서도 1×1 convolution 접근 방식 사용
  ```
  1×1 convolution
  - 차원 축소를 통한 계산 병목(Computational Bottleneck) 제거
  - 성능 저하 없이 네트워크 깊이와 너비 증가
  ```

<br>

- Object Detection 분야에서의 R-CNN:
  ```
  1. 색상이나 슈퍼픽셀 일관성 등 저수준 단서(Low-Level cues)로 카테고리에 구애받지 않는 방식(category-agnostic)으로 object 후보 생성
    -> 바운딩 박스 분할 정확도 향상
  2. CNN Classifier로 해당 위치의 객체 범주 식별
    -> CNN의 강력한 분류 성능 활용
  ```
- 본 논문에서도 Object model로 유사한 파이프라인을 사용
  ```
  추가 개선점 탐구
  - multi-box 예측 기법을 활용한 object bounding boxd의 recall 증가
  - 앙상블 접근 방식을 사용한 bounding box 후보 분류 성능 향상
  ```
<br>

## Motivation and High Level Considerations

<br>

- 심층 신경망 성능 향상을 가장 간단한 방법은 네트워크의 깊이와 너비를 포함한 네트워크의 크기를 늘리는 방법임

- 대규모의 라벨링된 학습 데이터가 존재하는 경우 적합하나 두 가지 주요 단점 가짐
```
1. 네트워크 크기 커질 수록 paprameter 수 증가 -> overfitting에 더 취약해짐
2. 네트워크 크기 증가할수록 자원 소모량이 급격히 증가함
```
<br>

- 문제 해결 방법 : Convolutions 내부에서 dense 한 Fully-Connected 구조를 Sparsely Connected 구조로 전환
![image](https://github.com/user-attachments/assets/0f698514-e565-443a-b8e4-f39d9314975b)


<br>

=> Inception 아키텍처는 희소 구조를 조밀한(Dense) 구성 요소들로 근사하여 구현하는 접근법임

<br>

## Architectural Details

<br>

#### 기존의 Inception module

- Inception 아키텍처의 핵심 아이디어 : CNN에서 각 요소를 최적의 Local Sparse Structure로 근사화하고 dense component로 변환하는 것 <br>
  -> Sparse 매트릭스를 서로 묶어(클러스터링 하여) 상대적으로 Dense 한 Submatrix를 만든다는 것
- 이전 layer의 각 유닛이 입력 이미지의 특정 부분에 해당한다고 가정했을때 lower layer(입력과 가까운 layer)에서는 특정 부분(인접한 유닛들) 강한 상관성을 보이므로 1×1 Convolutions으로 처리할 수 있음
  ![image](https://github.com/user-attachments/assets/fbc338a6-a2a7-4e2c-8dc7-f908ea701900)
  ```
  공간적으로 널리 퍼진(Spread-Out) 클러스터의 경우 조금 더 큰 patch 기반 Convolutions으로 처리 가능
  -> 영역 커질수록 patch의 개수는 점점 감소함
  ```

<br>

- 그림처럼 더 넓은 영역의 Convolution filter가 있어야 상관관계 높은 뉴런(Correlated unit)을 덮을 수 있는 경우가 있음
- 본 논문의 Inception 아키텍처에서는 편의를 위해 filter size를 1x1, 3x3, 5x5로 제한하여 다양한 크기의 filter를 조합함
- 또한 이미 성능이 검증된 pooling layer를 통해 병렬적인 pooling path를 추가하여 성능 향상시킴

  <br>

  ![image](https://github.com/user-attachments/assets/045daf1d-78c9-4daf-ac54-3d7b2dac20b2)
- Inception Modules은 layer을 쌓아 output에 가까워질수록 고차원의 특징을 학습하게 되면서 공간적 집중도(Spatial Concentration)는 감소하게 됨
- 그러므로 3×3 및 5×5 conv layer를 비율을 늘려야 하는데 이는 연산량 급증으로 이어짐

<br>

#### 새로운 버전의 Inception module

- 기존의 3×3 및 5×5 conv layer 비율 증가로 인한 연산량 급증을 줄이기 위해 1×1 conv layer 제외한 각각 conv layer 앞과 polling layer 뒤에 1×1 conv layer를 추가함
- 이를 통해 차원 축소(Dimension Reduction) 및 conv layer 뒤에 따라오는 ReLU 함수를 이용하여 연산량을 감소시킴
  ![image](https://github.com/user-attachments/assets/1771c79e-e92a-4d71-b73a-c367a6ce2e10)

- Inception 아키텍처의 이점은 각 층별 unit의 수가 크게 증가해도 연산량 유지가 가능함


<br>

## GoogLeNet

<br>

- 본 파트에선 Inception module이 적용된 GoogLeNet의 구조에 대해 설명함
  ![image](https://github.com/user-attachments/assets/27ec0f35-169c-4bc0-a44d-e7efb9e94b0c)

- 모든 Conv layer에는 ReLU가 적용됨
- 본 모델의 입력은 224×224의 RGB 이미지를 사용하며 Mean Subtraction(평균값 제외)를 적용함
- #3×3 reduce 및 #5×5 reduce는 3×3 및 5×5 conv layer 앞에 적용하는 1×1 filter 채널 수를 의미함
- pool proj 열은 max-pooing layer 뒤에 오는 1×1 filter 채널 수를 의미함

<br>

#### part A

![image](https://github.com/user-attachments/assets/22daba68-a920-465d-9c6f-fd96662ff295)

```
Part A : 입력 이미지와 가장 가까운 layer가 위치한 곳
연산 효율성을 위해 lower layer에는 기본적인 CNN 구조를 띄고 있음
```

<br>

#### part B

![image](https://github.com/user-attachments/assets/e00c6fd7-dcdd-4418-812b-58caab21b140)
```
Part B : Inception module로 다양한 feature 추출을 위해 1×1, 3×3, 5×5 conv layer 병렬 연산 수행함
차원 축소를 통해 연산량을 줄이고자 3×3, 5×5 conv layer 앞에 1×1 conv layer를 적용함
```

<br>

#### part C

![image](https://github.com/user-attachments/assets/7b92d7ac-6990-47b5-9c6f-6c86410fcc86)
```
Part C : auxiliary classifier가 적용된 부분
네트워크 깊이가 깊어질수록 Gradient Vanishing 문제가 발생할 수 있음
상대적으로 옅은 네트워크에서도 좋은 성능을 보이는 것을 통해 네트워크 중간 layer에서 생성된 feature가 매우 차별성이 강함을 시사함
중간 layer에서 auxiliary classifier를 추가하여 출력 결과를 역전파로 전달하여 gradient가 잘 전달되도록 돕고, 정규화 효과도 제공함

auxiliary classifier : 작은 CNN 형태를 뛰며 학습 중에는 auxiliary classifier의 loss에 가중치 0.3을 곱하고 추론 시 제거함
```

<br>

#### part D

![image](https://github.com/user-attachments/assets/25fe0117-795b-48d6-bc83-5cda031ffe32)
![image](https://github.com/user-attachments/assets/62aec2ab-3741-4653-b569-3f3250c941a0)

```
Part D : 예측 결과가 나오는 모델의 끝 부분
classifier 이전에 Global Average pooling layer(GAP)를 적용함
이전 layer에서 추출한 feature map들에 대한 평균을 낸 뒤 이어서 1차원 벡터로 변환

FC layer를 이용하여 1차원 벡티로 변환하는 경우 가중치 개수는 7x7x1024 = 51.3M임
but GAP 적용 시 단 한개의 가중치가 필요없으므로 평균내어 1차원 벡터로 만들면 가중치 개수를 상당히 줄여줌
```

<br>

![image](https://github.com/user-attachments/assets/d2504d91-1c61-4e7e-ae2b-59085e787a4b)

<br>

<br>

## Training Methodology

<br>

- SGD + Momentum : 0.9
- 8 Epochs마다 Learning Rate 4%씩 감소시킴
- Final Model은 Polyak Averaging 사용하여 생성
- 이미지 샘플링 방법
  - 다양한 크기의 patches 샘플링 : patch 크기는 원본 이미지의 8 ~ 100% 포함하도록 균등분포 샘플링
  - 이미지 비율은 3:4 ~ 4:3 사이에서 무작위 선택
  - Photometric Distortions 기법 사용
