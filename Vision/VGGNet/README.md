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

- 본 논문에선 3x3 filter(stride=1) 사용함
- 두 개의 3x3 conv layer 적용 시 5x5 conv layer와 동일하고 세 개의 3x3 conv layer은 7x7 layer와 동일함
- 그럼 왜 7x7 conv layer 한 번 적용하는 대신 3개의 3x3 conv layer를 적용하는가
  - ReLU를 더 많이 통과하여 비선형성을 증가시켜 모델이 더 강력한 분류 능력을 가질 수 있음
  - paramter의 개수가 감소하여 overfitting을 줄일 수 있음
    ```
    - 입출력 크기가 C개인 3x3 layer :
      3 x (3 x 3 x C^2) = 27C^2

    - 입출력 크기가 C개인 7x7 layer :
      7 x 7 x C^2 = 49C^2
    ```
<img src="https://github.com/user-attachments/assets/8bd86e80-3271-4172-a864-d0034f5d4e5c" width="500" height="500"/>
<img src="https://github.com/user-attachments/assets/a6a705cd-1cbc-42e3-97e7-151c785ae882" width="500" height="500"/>


- 본 논문에선 또한 1x1 conv layer 적용함
- 1x1 conv layer를 통해 receptive field(특정 convolution 뉴런이 보고 있는 입력 이미지 일부) size를 건들지 않고 비선형성을 증가시킴

<br>

##  CLASSIFICATION FRAMEWORK

<br>

- 해당 섹션에서는 classification ConvNet에 대한 training과 evaluation 을 설명함

<br>

### TRAINING

- ConvNet 훈련 절차는 기본적으로 AlexNet을 따름(multi-scale train image에서 input crop 샘플링 제외)
  - Optimization : multinomial Logistic Regression
  - mini-batch gradient descent with momentum
  - batch size : 256
  - momentum size : 0.9
  - Regularization : weight decay(L2 Norm with $5 × 10^{-4}$)
  - Dropout : 1,2번째 FC layer에서 0.5로 설정
  - Learning rate : 초기 0.01로 설정 validation accuracy가 개선되지 않을때마다 10배씩 감소 -> 최종적으로 3번 감소하여 0.00001로 끝남(74 epoch 후 학습 종료함)

<br>

- AlexNet보다 깊이가 더 깊고 parameter 수가 많음에도 더 적은 epoch 수로 수렴한 이유
  - network가 더 깊어지고 conv filter가 작어짐에 따라 발생하는 implicit regularization 효과
  - 특정 layer에 대한 pre-initialization 수행

<br>
 
- 본 논문에서는 잘못된 초기화를 막기 위해 아래와 같은 방법 사용
  - Configuration A를 먼저 학습한 뒤 더 깊은 network를 훈련할 때는 초기 4개의 conv layer와 마지막 3개의 fc layer를 network A의 weight로 초기화함, 나머지 중간 layer는 무작위로 초기화함
  - 무작위 초기화의 경우 weight는 평균은 0, 분산은 0.01인 정규분포에서 샘플링했고 bias는 0으로 초기화

<br>
 
- train image size는 224x224로 고정하며 원본 이미지에서 train image size에 맞게 무작위로 crop하여 샘플링함
- Data augmentation은 3가지 방법 사용
  - 무작위로 crop 한 이미지에 대해 무작위로 좌우 반전 적용
  - 무작위로 RGB 색 변환
  - image rescaling
    - Single scale : input image size를 고정된 크기로 scale함
    - Multi scale : 원본 이미지에서 정의한 input image size만큼 무작위로 샘플링함

<br>

### TESTING

- test image size를 다양하게 rescale 하여 입력값을 사용함
- 이를 위해 1번째 FC layer는 7x7 conv layer를 사용하고 나머지 2개의 FC layer는  1x1 conv layer를 사용함


<br>

## CLASSIFICATION EXPERIMENTS

<br>

### SINGLE SCALE EVALUATION

![image](https://github.com/user-attachments/assets/ae10f0c2-5bc7-46f8-a442-93c556840d56)


<br>

### MULTI-SCALE EVALUATION

![image](https://github.com/user-attachments/assets/1fbfe8b5-ac6a-4b05-ba7d-57f82151ffaf)


<br>

### MULTI-CROP EVALUATION

![image](https://github.com/user-attachments/assets/fdcdfa0b-fe89-44a6-b101-ac5e17f69200)


<br>

### CONVNET FUSION

![image](https://github.com/user-attachments/assets/08aebdaa-5cec-4a42-ba93-0b7d105e16c7)


<br>

### COMPARISON WITH THE STATE OF THE ART

![image](https://github.com/user-attachments/assets/9e194d81-8f4b-426f-9b27-c7fb5da32f64)

