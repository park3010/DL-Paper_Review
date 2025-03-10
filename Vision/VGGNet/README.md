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

- Input size : 224 x 224 size의 RGB image(3 channel) -> input image는 3x3 크기의 filter 사용
- preprocessing : 각 pixel에 대해 training set에서 계산된 RGB 평균을 뺌
- Convolution 연산 : stride = 1 고
- MaxPooling : stride = 2, 2x2 pixel 사이즌에서 window 수행됨
- Convolution layer 이후엔 3개의 Fully Connected layer가 위치함
  - 1,2 FC layer : 4096 channel
  - 3 FC layer : 1000 channel
  - last layer : softmax layer
- ReLU 함수 사용

<br>

### CONFIGURATIONS

![image](https://github.com/user-attachments/assets/7b330852-e1ea-4c0c-80d9-5b2e3e6cdd0a)

<br>

- network A는 11개의 weight layer(conv 8개, fc 3개), network E는 19개의 weight layer(conv 16개, fc 3개) 로 **깊이** 만 다르게 구성함
- 첫 번째 layer에서 channel 수는 64개 -> MaxPooling layer 이후에는 512개의 channel로 증가함

<br>

![image](https://github.com/user-attachments/assets/53e0d720-c36d-4148-8178-065e61e94880)

<br>

- network A ~ E 까지의 parameter 개수 정리
- network의 깊이가 깊어짐에도 해당 network보단 깊이는 얕지만 더 넓은 Conv layer와 filter를 가진 다른 network보단 parameter 수가 적음


<br>

### DISCUSSION

- 

