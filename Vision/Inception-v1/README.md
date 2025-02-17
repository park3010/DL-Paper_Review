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
  - 대표적인 예제로 Girshick et al의 R-CNN이 있음<br>
  
- 효율성을 고려하여 모델이 추론(Inference) 시 약 15억 번의 곱셈-덧셈 연산 내에서 계산 비용 유지하도록 설계함<br>

- 컴퓨터 비전을 위한 효율적인 심층 신경망 : Inception
  - 본 논문에서 본 "Deep"의 의미
    - “Inception module" 이라는 새로운 organization(조직화) 방식 도입한다는 점에서 깊음
    - 보다 직접적으로 네트워크 깊이를 증가시킴
  - Lin et al의 연구를 확장한 형태이며, Arora et al의 이론적 연구에서 영감을 받음

<br>

## Related Work

<br>

- CNN은 일반적으로 연속적으로 쌓은 Convolution layer 층(선택에 따라 대비 정규화(contrast normalization)나 Maxpooling이 따라오기도 함)이 배치되어 있으며 하나 이상의 Fully-connected layer가 배치되는 표준 형태를 따름
- 데이터 데이터 셋 처리 트랜드 -> layer 수 증가, layer의 크기 증가, dropout 기법을 활용한 overfitting 방지
- Serre et al은 영장류 시각 피질(Primate Visual Cortex)의 신경 과학 모델에 영감을 받아 다양한 크기의 고정된 Gabor Filter를 활용한 multiple scale 문제 처리 방식을 제안함
  - 해당 방식은 Inception가 유사하나 차이점 존재함
    - Serre et al은 고정된 2-layer 모델을 사용함
    - Inception 모델은 모든 필터가 학습되며








