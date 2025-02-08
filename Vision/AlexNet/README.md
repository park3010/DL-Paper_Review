# ImageNet Classification with Deep Convolutional Neural Networks

<br>

```plaintext
한 줄 요약 : 
```

<br>

## Abstract

<br>

- 본 논문에서 제안하는 AlexNet을 사용하여 ImageNet LSVRC-2010 대회에서 12만 개의 고해상도 이미지를 1000개의 class를 분류함
  - top-1, top-5에서 각각 오차율 37.5%와 17.0%를 달성함
  - 이 수치는 기존 최첨단 모델보다 좋은 수치임
- neural network 구성
  - 6천만 개의 parameter와 65만 개의 뉴런으로 구성됨
  - max-pooling layer를 동반한 5개의 convolution layer와 마지막에 1000개의 class 구분하는 softmax 적용한 fully-connected layer 3개로 구성
- 학습 속도 향상을 위해 제안한 방법 -> 효율적인 convolution 연산을 위한 GPU 방법과 non-saturating neurons을 사용함
- fully-connnected layer에서의 overfitting 감소 -> "dropout" 정규화 기법 사용함
- ILSVRC-2012 대회에서 변형된 모델을 제출하여 오차율 26.2%를 달성한 2위 모델보다 뛰어난 15.3%를 달성하여 우승함

<br>
