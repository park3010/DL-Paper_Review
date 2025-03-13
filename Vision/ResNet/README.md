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

- 본 논문에서는 Residual Learning Framework를 제안함
  ![image](https://github.com/user-attachments/assets/2d04bc71-c895-41c1-9251-72d032c0347c)
  ```
   [residual train]
  입력값 x를 받아 타겟값 y로 mapping하는 H(x)를 학습하고자 할 때

  - 기존 layer : 입력 x를 받아 layer를 거쳐 H(x)를 출력함
  - ResNet layer : layer가 H(x)를 직접 학습하는 것이 아닌 출력과 입력의 잔차인 F(x) = H(x) - x 를 학습하도록 설계함
    -> 즉, F(x) = H(x) - x를 원래 학습하고자 했던 함수 H(x) = F(x) + x로 변환 가능

  - F(x) = H(x) - x를 학습하는 만큼 이를 최소화시켜야 하며, 이는 출력과 입력의 차를 줄여야 함을 의미함
  - F(x) = 0이 최적의 해가 되며 0 = H(x) - x, H(x) = x로 mapping하는 것이 학습 목표가 됨

  - 이러한 잔차 학습(residual train)은 shortcut connection 형태로 구현됨
  ```
