---
layout: single
title:  "TensorFlow 초보자용 튜토리얼"
categories: ML-and-DL
tags: [python, tensorflow, ML, DL]
author_profile: false
toc: true
toc_sticky: true
toc_label: 목차
---

# 1. TensorFlow 설정하기


```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
```

    TensorFlow version: 2.12.0
    

# 2. 데이터셋 로드하기


```python
mnist = tf.keras.datasets.mnist # keras API 사용 MNIST 데이터셋 로드

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

MNIST는 손으로 쓴 숫자 이미지 데이터셋으로, 딥러닝을 학습하고 테스트하는 데 사용, 0부터 9까지의 숫자를 포함하는 28 x 28 크기의 흑백 이미지로 구성 <br/>

학습 데이터와 테스트 데이터를 0에서 255까지의 값으로 정규화 <br/>
각 픽셀의 값을 0에서 1사이로 조정하여 모델이 더 잘 학습되도록 돕는 일반적인 전처리 단계

# 3. 머신 러닝 모델 빌드


```python
# 숫자 인식을 위한 간단한 신경망 모델을 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)), # 28x28 픽셀의 2D 배열에서 1D배열로 변환
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation = 'softmax')
])
```

**tf.keras.models.Sequential**은 순차적인 모델을 생성하기 위한 클래스입니다. 순차적인 모델은 레이어를 순차적으로 쌓아나가는 방식으로 구성됩니다. <br/>
**tf.keras.layers.Flatten(input_shape = (28, 28))**: 이 레이어는 28 x 28 크기의 이미지를 1차원 벡터로 펼치는 작업을 수행합니다. 784(=28x28)길이의 1차원 벡터로 변환합니다. 이렇게 하는 이유는 다음 레이어에서 사용될 완전 연결 레이어에 입력하기 위해서입니다. <br/>
<br/>
**tf.keras.layers.Dense(128, activation = 'relu')**: 128개의 뉴런으로 구성된 완전 연결 레이어입니다. 이 레이어는 ReLU(Rectified Linear Unit)활성화 함수를 사용하여 입력 데이터에 비선형성을 도입합니다. ReLU는 입력이 양수인 경우에는 그 값을 그대로 출력하고, 음수인 경우엔 0을 출력하는 함수. <br/>
<br/>
**tf.keras.layers.Dropout(0.2)**: 입력 데이터의 20%를 무작위로 0으로 만듭니다. 이는 과적합(overfitting)을 방지하기 위한 정규화(regularization) 기법 중 하나로, 모델이 훈련 데이터에 과도하게 적합되는 것을 방지하여 일반화 성능을 향상시킵니다. <br/>
<br/>
**tf.keras.layers.Dense(10, activation = 'softmax')**: 10개의 뉴런으로 구성된 완전 연결 레이어입니다. MNIST 데이터셋의 10개의 클래스(0부터 9까지의 숫자)를 예측하는 분류 문제를 다루기 때문에 출력 뉴런의 개수는 10입니다. 또한, 출력에 대해 소프트맥스(softmax)활성화 함수를 사용하여 각 클래스에 대한 확률 분포를 출력합니다. 소프트맥스 함수는 출력 벡터의 각 원소를 0과 1사이의 값으로 정규화하고, 이 값들의 합이 1이 되도록 만듭니다. 이를 통해 모델은 각 클래스에 속할 확률을 추정할 수 있습니다.


```python
# 모델을 컴파일, 학습 과정에서 사용할 옵티마이저, 손실 함수, 평가 지표 설정
model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])
```

**optimizer = 'adam'**: 옵티마이저는 모델이 손실 함수를 최소화하기 위해 사용하는 최적화 알고리즘을 의미합니다. 여기서는 'adam' 옵티마이저를 선택했습니다. Adam은 학습 속도를 조절하는 방법으로 널리 사용되는 옵티마이저입니다. <br/>
<br/>
**loss = 'sparse_categorical_crossentropy'**: 손실 함수는 모델이 학습 중에 얼마나 잘 예측하는지를 측정하는 데 사용됩니다. 'sparse_categorical_crossentropy'는 다중 클래스 분류 문제에 적합한 솔실 함수입니다. 이 함수는 정수 형태로 레이블된 클래스를 처리할 수 있으며, 각 예측과 실제 레이블 간의 차이를 계산하여 손실 값을 얻습니다. <br/>
<br/>
**metrics = ['accuracy']**: 평가 지표는 모델의 성능을 평가하는 데 사용되는 지표입니다. 여기서는 'accuracy'를 사용했습니다. 'accuracy'는 정확도를 의미하며, 예측 결과가 실제 레이블과 얼마나 일치하는지를 측정합니다. 이를 통해 모델이 얼마나 정확하게 분류하는지를 평가할 수 있습니다.


```python
# 첫 번째 이미지에 대한 모델의 예측 수행
predictions = model(x_train[:1]).numpy() # numpy(): 예측 결과를 NumPy 배열로 변환
predictions
```




    array([[0.05608082, 0.06618251, 0.04057353, 0.14124723, 0.0666269 ,
            0.21558239, 0.07158391, 0.08923257, 0.173829  , 0.07906111]],
          dtype=float32)




```python
# 모델의 예측 결과에 softmax 함수 적용하여 확률 분포로 변환
tf.nn.softmax(predictions).numpy()
```




    array([[0.09555908, 0.09652928, 0.09408864, 0.10405412, 0.09657219,
            0.11208374, 0.09705208, 0.09878013, 0.10750022, 0.09778048]],
          dtype=float32)



**tf.nn.softmax(predictions)**는 'predictions' 배열에 softmax 함수를 적용하는 TensorFlow의 함수입니다. softmax 함수는 각 원소의 값을 0과 1사이로 정규화하고, 이 값들의 합이 1이 되도록 만들어 확률 분포로 변환합니다.


```python
# 로짓의 벡터와 True 인덱스 사용 각 예시에 대해 스칼라 손실을 반환하는 훈련용 손실 함수 정의
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
```

**tf.keras.losses.SparseCategoricalCrossentropy**는 다중 클래스 분류 문제에 사용되는 손실 함수입니다. 이 함수는 실제 정수 형태로 레이블된 클래스와 모델의 예측값 사이의 차이를 계산하여 손실을 측정합니다. 이 손실 함수는 클래스 인덱스를 인코딩할 필요 없이 정수 형태의 레이블을 바로 처리할 수 있습니다. <br/>
<br/>
**from_logits = True**는 모델의 출력값이 확률 분포로 정규화되지 않은 'logits'형태인 경우에 사용됩니다. 'logtis'는 확률을 나타내기 전에 softmax 함수를 거치지 않은 모델의 출력값을 의미합니다. 따라서 이렇게 설정하면 손실 함수 내부에서 softmax 함수를 적용하여 확률 분포로 변환하는 과정이 수행됩니다.


```python
# 첫 번째 이미지 손실 값 계산
loss_fn(y_train[:1], predictions).numpy()
```




    2.188509




```python
model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = ['accuracy'])
```

# 4. 모델 훈련 및 평가하기


```python
model.fit(x_train, y_train, epochs= 5) # 5번 반복하여 모델 학습
```

    Epoch 1/5
    

    c:\Users\YONSAI\anaconda4\lib\site-packages\keras\backend.py:5612: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
      output, from_logits = _get_logits(
    

    1875/1875 [==============================] - 5s 3ms/step - loss: 0.2945 - accuracy: 0.9138
    Epoch 2/5
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1413 - accuracy: 0.9574
    Epoch 3/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.1071 - accuracy: 0.9674
    Epoch 4/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0887 - accuracy: 0.9725
    Epoch 5/5
    1875/1875 [==============================] - 5s 3ms/step - loss: 0.0749 - accuracy: 0.9757
    




    <keras.callbacks.History at 0x1ac63ce6fa0>




```python
model.fit(x_train, y_train, epochs = 5)
model.evaluate(x_test, y_test, verbose = 2) # verbose: 평가 과정의 상세도를 조절하는 매개변수
```

    Epoch 1/5
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.0663 - accuracy: 0.9786
    Epoch 2/5
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.0581 - accuracy: 0.9816
    Epoch 3/5
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.0531 - accuracy: 0.9831
    Epoch 4/5
    1875/1875 [==============================] - 5s 2ms/step - loss: 0.0490 - accuracy: 0.9832
    Epoch 5/5
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.0443 - accuracy: 0.9857
    

    c:\Users\YONSAI\anaconda4\lib\site-packages\keras\backend.py:5612: UserWarning: "`sparse_categorical_crossentropy` received `from_logits=True`, but the `output` argument was produced by a Softmax activation and thus does not represent logits. Was this intended?
      output, from_logits = _get_logits(
    

    313/313 - 1s - loss: 0.0695 - accuracy: 0.9790 - 504ms/epoch - 2ms/step
    




    [0.06954585015773773, 0.9789999723434448]




```python
# 기존 모델에 소프트맥스 함수를 적용하여 확률 출력하는 새로운 모델
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])
```


```python
probability_model(x_test[:5])
```




    <tf.Tensor: shape=(5, 10), dtype=float32, numpy=
    array([[0.08533742, 0.08533742, 0.08533742, 0.08534161, 0.08533742,
            0.08533742, 0.08533742, 0.23195864, 0.08533742, 0.08533781],
           [0.08533704, 0.08533896, 0.23196459, 0.08533712, 0.08533704,
            0.08533706, 0.08533704, 0.08533704, 0.08533704, 0.08533704],
           [0.08533964, 0.23192368, 0.08534087, 0.08533964, 0.08533993,
            0.08533964, 0.08533966, 0.08535735, 0.08534005, 0.08533964],
           [0.23194201, 0.08533847, 0.08533958, 0.08533847, 0.08534563,
            0.0853387 , 0.08533931, 0.08534022, 0.08533847, 0.08533912],
           [0.08537383, 0.08537382, 0.08537399, 0.08537382, 0.23138164,
            0.08537382, 0.08537382, 0.08537462, 0.08537383, 0.08562685]],
          dtype=float32)>


