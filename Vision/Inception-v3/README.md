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

