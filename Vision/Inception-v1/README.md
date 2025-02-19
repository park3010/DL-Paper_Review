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
- 그러므로 3×3 및 5×5 conv layer를 비율을 늘려야 하는데 이는 연산량 급증으로 이어짐 <br>
  -> 
