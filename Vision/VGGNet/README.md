# VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION

<br>

## ABSTRACT

<br>

- 대규모 이미지 인식 설정에서 Convolution network의 깊이가 정확도에 미치는 영향을 조사함
- 굉장히 작은 3x3 conv filter를 사용하여 네트워크 깊이에 따른 성능 평가하여 16~19 weight layer에서 성능 개선됨을 발견함
- ConvNet 구조를 개선하여 더 개선된 ConvNet을 개발했으며 ILSVRC의 classification task와 localisation task에서 SOTA를 기록함

<br>

##  CONVNET CONFIGURATIONS

<br>

- 해당 섹션에선 ConvNet의 generic layout의 구성을 설명하고 모델 평가를 위해 사용된 구성을 설명함


<br>

### ARCHITECTURE

![image](https://github.com/user-attachments/assets/04aabecd-c2a6-44bf-92a0-90a6ecd61b31)

<br>

- Input size : 224 x 224 size의 RGB image(3 channel)
- preprocessing : 각 pixel에 대해 training set에서 계산된 RGB 평균을 뺌
- 
