---
layout: single
title:  "PyTorch Quickstart로 연습하기"
categories: ML-and-DL
tags: [python, pytorch, ML, DL]
author_profile: false
toc: true
toc_sticky: true
toc_label: 목차
---

# 1. Working with data


```python
import torch
from torch import nn
from torch.utils.data import DataLoader # Dataset을 둘러싸고 반복 가능한 객체로 만듦
from torchvision import datasets # 샘플과 해당하는 레이블 저장
from torchvision.transforms import ToTensor
```

PyTorch는 TorchText, TorchVision, TorchAudio와 같은 도메인별 라이브러리를 제공하는데 이들은 모두 데이터셋을 포함하고 있습니다. 이 튜토리얼에서는 TorchVision 데이터셋을 사용할 것입니다. <br/>
<br/>

torchvision.datasets 모듈은 CIFAR, COCO와 같은 다양한 실제 비전 데이터에 대한 Dataset 객체를 포함하고 있습니다(전체 목록은 여기에서 확인할 수 있습니다.). 이 튜토리얼에서는 FashionMNIST 데이터셋을 사용합니다. TorchVision 데이터셋은 transform과 target_transform 두 가지 인수를 포함하고 있어, 각각 샘플과 레이블을 수정하는 데 사용됩니다.


```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root = "data", # 데이터 저장 경로 지정
    train = True, # 훈련 데이터를 다운로드합니다.
    download = True, # 데이터를 인터넷에서 다운로드합니다.
    transform = ToTensor(), # 데이터를 텐서 형태로 변환합니다.
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root = "data",
    train = False,
    download = True,
    transform = ToTensor(),
)
```

    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz to data/FashionMNIST/raw/train-images-idx3-ubyte.gz
    

    100%|██████████| 26421880/26421880 [00:00<00:00, 117538735.72it/s]
    

    Extracting data/FashionMNIST/raw/train-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
    

    100%|██████████| 29515/29515 [00:00<00:00, 9389069.59it/s]
    

    Extracting data/FashionMNIST/raw/train-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
    

    100%|██████████| 4422102/4422102 [00:00<00:00, 64795928.37it/s]
    

    Extracting data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz to data/FashionMNIST/raw
    
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
    Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
    

    100%|██████████| 5148/5148 [00:00<00:00, 16333038.57it/s]

    Extracting data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz to data/FashionMNIST/raw
    
    

    
    

Dataset을 DataLoader의 인수로 전달합니다. 이는 데이터셋을 반복 가능한(iterable) 객체로 감싸며, 자동 배치, 샘플링, 셔플링 및 다중 프로세스 데이터 로딩을 지원합니다. 여기에서는 배치 크기를 64로 정의하였으므로, 데이터로더의 각 요소는 64개의 특징(feature)과 레이블(label)로 구성된 배치를 반환할 것입니다.


```python
batch_size = 64 # 배치 크기

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size = batch_size)
test_dataloader = DataLoader(test_data, batch_size = batch_size)

for X, y in test_dataloader:
  print(f"Shape of X [N, C, H, W]: {X.shape}")
  print(f"Shape of y: {y.shape} {y.dtype}")
  break
```

    Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
    Shape of y: torch.Size([64]) torch.int64
    

배치 크기는 데이터를 미니배치(minibatch)로 나누는 데 사용되는 값입니다. 데이터를 한 번에 모두 처리하는 것이 아니라 작은 미니배치로 나누어 처리하면 몇 가지 이점이 있습니다.
- 메모리 효율성: 대용량 데이터셋의 경우 전체 데이터를 한 번에 메모리에 로드하는 것은 메모리 부담이 될 수 있습니다. 배치 크기를 작게 설정하면 각 배치만큼의 메모리만 사용하므로 메모리 사용이 효율적입니다.
- 가속화된 학습: GPU를 사용하는 경우, 배치 처리를 통해 병렬 계산을 수행할 수 있습니다. 배치 크기가 클수록 GPU의 병렬 처리 능력을 최대한 활용할 수 있습니다. 이를 통해 학습 속도를 향상시킬 수 있습니다.
- 일반화 능력 향상: 미니배치를 사용하면 데이터의 다양성을 확보할 수 있습니다. 다양한 데이터 샘플이 포함된 미니배치를 사용하면 모델이 보다 일반화된 패턴을 학습할 수 있습니다.
<br/>

또한, 배치 크기는 하이퍼파라미터로서 조정이 가능하며, 최적의 배치 크기는 문제와 데이터에 따라 다를 수 있습니다. 일반적으로는 실험과 검증을 통해 적절한 배치 크기를 찾게 됩니다.

# 2. Creating Models
PyTorch에서 신경망을 정의하기 위해, nn.Module을 상속한 클래스를 생성합니다. 신경망의 층(layer)들은 init 함수에서 정의하고, 데이터가 신경망을 통과하는 방식은 forward 함수에서 지정합니다. 신경망 연산을 가속화하기 위해, GPU나 MPS(모델 병렬 처리)가 사용 가능한 경우에는 해당 장치로 신경망을 이동시킵니다.


```python
# Get cpu, gpu or mps device for training.
device = (
    "cuda" # GPU가 사용 가능한 경우 CUDA 장치 선택
    if torch.cuda.is_available()
    else "mps" # MPS가 사용 가능한 경우 MPS 장치 선택
    if torch.backends.mps.is_available()
    else "cpu" # 그 외의 경우 CPU 장치 선택
)
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # 2D 이미지를 1D로 평탄화
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512), # 입력 크기: 28*28, 출력 크기: 512
            nn.ReLU(), # ReLU 활성화 함수
            nn.Linear(512, 512), # 입력 크기: 512, 출력 크기: 512
            nn.ReLU(), # ReLU 활성화 함수
            nn.Linear(512, 10) # 입력 크기: 512, 출력 크기: 10 (클래스 개수)
        )

    def forward(self, x):
        x = self.flatten(x) # 입력을 평탄화
        logits = self.linear_relu_stack(x) # 평탄화된 입력을 신경망에 통과
        return logits

# 모델 인스턴스 생성 및 장치로 이
model = NeuralNetwork().to(device)
print(model)
```

    Using cpu device
    NeuralNetwork(
      (flatten): Flatten(start_dim=1, end_dim=-1)
      (linear_relu_stack): Sequential(
        (0): Linear(in_features=784, out_features=512, bias=True)
        (1): ReLU()
        (2): Linear(in_features=512, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=10, bias=True)
      )
    )
    

# 3. Optimizing the Model Parameters
모델을 학습시키기 위해서는 손실 함수(loss function)와 옵티마이저(optimizer)가 필요합니다.


```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
```

- **loss_fn**: 손실 함수로 CrossEntropyLoss를 사용합니다. 이 함수는 다중 클래스 분류 문제에 적합한 손실 함수입니다.
- **optimizer**: 옵티마이저로 SGD(Stochastic Gradient Descent)를 사용합니다. 모델의 파라미터를 업데이트하기 위해 경사하강법을 활용합니다. 학습률(learning rate)은 1e-3로 설정되었습니다.

한 번의 학습 루프에서 모델은 학습 데이터셋에 대한 예측을 수행하고 (배치 단위로 제공됨), 예측 오차를 역전파하여 모델의 파라미터를 조정합니다.


```python
def train(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset) # 데이터셋의 총 샘플 개수
  model.train() # 모델을 학습 모드로 설정
  for batch, (X, y) in enumerate(dataloader):
    X, y = X.to(device), y.to(device) # 데이터와 레이블을 GPU로 이동

    # 예측 오차 계산
    pred = model(X) # 모델을 사용하여 예측 수행
    loss = loss_fn(pred, y) # 예측과 실제 레이블 사이의 손실 계산

    # 역전파
    loss.backward() # 손실에 대한 역전파를 수행
    optimizer.step() # 옵티마이저를 사용하여 모델의 파라미터 업데이트
    optimizer.zero_grad() # 모델의 변화도를 초기화

    if batch % 100 == 0:
      loss, current = loss.item(), (batch + 1) * len(X)
      print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
```

모델의 성능을 확인하기 위해 테스트 데이터셋을 사용하여 모델을 평가합니다.


```python
def test(dataloader, model, loss_fn):
  size = len(dataloader.dataset) # 테스트 데이터셋의 샘플 개수
  num_batches = len(dataloader) # 배치의 개수
  model.eval() # 모델을 평가 모드로 설정
  test_loss, correct = 0, 0 # 테스트 손실과 정확도 초기화
  with torch.no_grad():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device) # 데이터를 디바이스(GPU 또는 CPU)로 이동
      pred = model(X) # 모델에 입력을 전달하여 예측 수행
      test_loss += loss_fn(pred, y). item() # 손실 값을 누적
      correct += (pred.argmax(1) == y).type(torch.float).sum().item() # 정확한 예측 수를 누적
  test_loss /= num_batches # 평균 손실 계산
  correct /= size # 정확도 계싼
  print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}% Avg loss: {test_loss:>8f} \n")
```

훈련 과정은 여러 번의 반복(epoch)을 거쳐 진행됩니다. 각 에포크에서 모델은 더 나은 예측을 위해 매개변수를 학습합니다. 각 에포크마다 모델의 정확도와 손실을 출력합니다. 정확도는 증가하고 손실은 감소하는 것을 기대합니다.

# 4. Saving Models
모델을 저장하는 일반적인 방법은 내부 상태 딕셔너리(모델 매개변수를 포함하는)를 직렬화하는 것입니다.


```python
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
```

    Saved PyTorch Model State to model.pth
    

# 5. Loading Models
모델을 로드하는 과정은 모델 구조를 재생성하고 그 상태 사전을 모델에 로드하는 것을 포함합니다.


```python
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))
```




    <All keys matched successfully>




```python
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval() # 모델을 평가 모드로 설정합니다.
x, y = test_data[0][0], test_data[0][1] # 테스트 데이터에서 예시 데이터를 가져옵니다.
with torch.no_grad():
  x = x.to(device) # 데이터를 디바이스(GPU 또는 CPU)로 이동합니다.
  pred = model(x) # 모델에 데이터를 전달하여 예측 수행
  predicted, actual = classes[pred[0].argmax(0)], classes[y] # 예측 결과와 실제 결과를 가져옴
  print(f'Predicted: "{predicted}", Actual: "{actual}"') # 예측 결과와 실제 결과 출력
```

    Predicted: "Sneaker", Actual: "Ankle boot"
    
