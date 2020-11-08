--- 
title: [CNN]VGG_net
use_math: true
comments: true
categories:
  - 딥러닝기초
  - CNN
---


## VGGNET

![vgg](http://whdbfla6.github.io/assets/images/vgg1.jpg)

**VGG16**은 K. Simonyan and A. Zisserman에 의해 제안된 convolution neural network 중 하나다. 이 모델은 1000개의 class로 구분되는 1400만장의 이미지를 분류하는데 높은 정확성을 보였다. VGG NET의 큰 특징은 1) 기존의 모델에 비해 depth가 증가했다 2)모든 Convolution layer에서 kernel size가 3인 필터를 사용했다는 점이다.

**1) Increasing depth**<br/>
VGG 연구팀은 Layer 깊이에 변화를 주어 여러 구조의 성능을 비교하였고, 깊이가 11 13 16 19층으로 가면서 에러가 감소하는 것을 확인하였다. 층이 16개인 경우를 VGG16 19개인 경우를 VGG19라 한다

**2) using 3x3 convolution filters**<br/>
VGG NET이 3x3 사이즈 필터를 사용한 이유는 다음과 같다. 3x3필터를 2개 사용하는 것이 5x5 필터를 한번 사용하는 것과 같은 효과라고 한다. 이렇게 되면 연산에 사용되는 파라미터의 수가 감소하게 되는데, 이는 오버피팅을 방지해준다.


## VGGNET 구조

구조는 VGGNET 모델을 직접 코드로 작성해보면서 설명하려고 한다

![vgg](http://whdbfla6.github.io/assets/images/vgg3.JPG)


이는 VGG11 VGG13 VGG16 VGG19의 구조를 나타낸 표다. D는 conv layer를 13번, FC를 3번 통과하기 때문에 VGG16에 해당한다.  VGG NET은 모두 마지막에 FC 3번 softmax를 통과하며, 앞부분에서 conv layer를 몇 번 통과시키는지에 따라 구조가 달라진다. 아래의 코드는 conv의 수를 조정하기 위한 부분이다.


```python
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], #8 + 3 =11 == vgg11
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], # 10 + 3 = vgg 13
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], #13 + 3 = vgg 16
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], # 16 +3 =vgg 19
}
```

아래와 같이 D(VGG16)을 예시로 들어 코드를 하나하나 살펴보겠다. `'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']`가 VGG16인 이유는 아래 설명을 따라가다 보면 이해할 수 있을 것이다


```python
conv = make_layers(cfg['D'], batch_norm=True) #VGG16
```


```python
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
                     
    return nn.Sequential(*layers)
```

`for v in cfg`에서 v는 크게 **숫자**와 **'M'**으로 구분할 수 있다.

> v가 숫자인 경우

숫자인 경우 `batch_norm=True` `v != 'M'`조건을 만족하기 때문에 `layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]`의 코드를 수행할 것이다. 여기서 각 숫자는 out_channels를 의미하는데, 첫번째 case는 64이므로 conv의 out_channel이 64다.

> v가 M인 경우

'M'이 들어오면 `layers += [nn.MaxPool2d(kernel_size=2, stride=2)]`코드를 수행하게 되는데, 이는 사이즈2 max-pooling을 의미한다.

- 즉 cfg에서 숫자 부분은 CONV,배치정규화,RELU 그리고 'M'은 max-pooling을 나타내며 `cfg['D']`는 숫자가 총 13개 + 이후의 FC 3번(이는 Class VGG에서 정의할 것이다)으로 **VGG16** 네트워크다.


```python
conv
```




    Sequential(
      (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU(inplace=True)
      (3): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (4): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (5): ReLU(inplace=True)
      (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (7): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (8): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (9): ReLU(inplace=True)
      (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (11): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (12): ReLU(inplace=True)
      (13): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (14): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (15): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (16): ReLU(inplace=True)
      (17): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (18): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (19): ReLU(inplace=True)
      (20): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (21): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (22): ReLU(inplace=True)
      (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (24): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (25): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (26): ReLU(inplace=True)
      (27): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (28): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (29): ReLU(inplace=True)
      (30): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (31): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (32): ReLU(inplace=True)
      (33): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      (34): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (35): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (36): ReLU(inplace=True)
      (37): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (38): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (39): ReLU(inplace=True)
      (40): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (41): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (42): ReLU(inplace=True)
      (43): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )



최종적인 VGG네트워크 구조를 나타내는 코드는 아래에 있다.


```python
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        
        self.features = features #convolution
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )#FC layer
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x) #Convolution 
        x = self.avgpool(x) # avgpool
        x = x.view(x.size(0), -1) #
        x = self.classifier(x) #FC layer
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```


```python
CNN = VGG(make_layers(cfg['D']), num_classes=10, init_weights=True)
```

> forward(self, x)

1) `self.features(x)` : 앞서 정의한 `conv`를 수행하는 부분<br/>
2) `self.avgpool(x)` : average-pooling<br/>
3) `self.classifier(x)` : Fully Connected layer

> initialize_weights(self)

`init_weights=True`을 지정해 주는 경우에 수행하며, weight들을 초기화하는 부분이다

최종적인 VGG16 구조는 아래와 같다


```python
CNN
```




    VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=10, bias=True)
      )
    )



설명을 위해서 코드 순서를 뒤죽박죽 나열했는데, 정돈된 코드는 [링크](https://github.com/deeplearningzerotoall/PyTorch/blob/master/lab-10_5_1_Advance-cnn(VGG).ipynb)에 있다.

참고문헌<br/>
[Neurohive-VGG16](https://neurohive.io/en/popular-networks/vgg16/)<br/>
[89douner](https://89douner.tistory.com/61)
