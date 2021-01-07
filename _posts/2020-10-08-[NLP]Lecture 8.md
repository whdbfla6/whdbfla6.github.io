--- 
title: '[NLP]Lecture 8'
usemathjax: true
use_math: true
comments: true
layout: single
classes: wide
categories:
  - NLP

---

본 글은 스탠포드 [CS 224N](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/) [Machine Translation,Sequence to sequence, and Attention] 강의를 들으며 정리한 글 입니다.

## 1. Statistical Machine Translation(SMT)

### 1) Machine Translation 이란?

**기계번역(Machine translation)** 은 x 문장을 다른 언어의 문장 y로 번역하는 것을 의미한다. 이 때 x를 *source language*의 문장, y를 *target language*의 문장이라 한다.

### 2) 이전의 Machine Translation(1950)

1950년대에 Machine translation 연구가 시작되었다. 번역은 규칙을 기반으로 이루어졌으며, 사전을 통해 대응하는 단어를 찾는 방식이었다. 

### 3) Statistical Machine Translation(1960-70)

- 핵심 아이디어: 데이터로부터 통계적 모델을 학습하는 것 

프랑스어를 영어로 번역한다고 하자. 프랑스어 문장이 주어졌을 때 가장 잘 번역된 영어 문장 y를 구하고 싶을 것이다. 이는 조건부 확률(x가 주어졌을 때 y가 나올 확률)을 최대화하는 y를 구하는 과정과 같다. 수식으로 표현하면 아래와 같다. 

$$argmax_yP(y|x)$$

- 이는 베이즈 정리에 따라 두개의 요소로 분해할 수 있다

$$= argmax_yP(x|y)P(y)$$

이 수식을 계산하는데 모든 y에 대해서 확률 값을 계산한다면 비용이 많이 들 것이다. 따라서 **heuristric search 알고리즘**을 사용해서 너무 낮은 확률값을 갖는 y는 배제하는 방식으로 진행된다.

> $P(y)$: Language model

단일어 사전으로부터 y문장(sequence of words)이 나올 확률을 구하는 부분. 

> $ P(x\mid y)$: Translation model

쌍으로 구성된 데이터로부터 source language와 target language의 번역 모델을 만드는 부분으로, $P(x,a\mid y)$ 을 고려한다. 여기서 a는 *alignment*로 **source langauge 단어들과 target 단어들의 상응하는 관계**를 의미한다.
![](https://i.imgur.com/AVR7pBn.png)
Alignment는 하나의 프랑스어가 여러개의 영어 단어로 표현되는 one-to-many관계 외에도, many-to-one/ many-to-many/ no counterpart 등 다양한 관계를 갖는다. 번역모델은 단어 간의 상응하는 관계, 문장에서의 위치, 하나의 단어가 몇개의 단어로 표현되는가 등 여러가지 요소를 고려한다. 

- 단점

SMT는 매우 복잡한 시스템이며 언어의 쌍으로 구성된 data를 구축하는데 많은 비용이 든다. 


## 2. Sequence to sequence

### 1) Neural Machine Translation(NMT)

NMT는 single neural network를 기반으로 한 기계번역 방법으로, sequence to sequence는 NMT 중 하나다. 

### 2) Sequence to sequence 구조

Sequence to sequence는 두개의 RNN이 연결된 구조로, 하나는 Encoder RNN 다른 하나는 Decoder RNN이라 한다.

![](https://i.imgur.com/BJzbXPz.png)


> 인코더 RNN : source sentence의 인코딩 작업이 이루어지는 부분

source language 문장의 워드 임베딩이 인풋 값으로 들어가며, RNN구조에 따라 마지막 hidden state가 나오게 된다. 인코더 RNN의 마지막 hidden state는 디코더 RNN의 Initial hidden state 역할을 한다. 

> 디코더 RNN : encoding이 주어진 상태에서 target sentence를 만들어내는 부분

인코더 RNN에서 생성된 마지막 Hidden state와 start 토큰을 받아서 예측값을 출력하고, 이 예측값을 다음 layer의 input값으로 들어가게 된다. 이 과정을 END토큰이 나올 때 까지 반복한다. 

> conditional language model 

target sentence의 첫번째 단어인 $y_1$이 나올 확률을 구하기 위해 source sentence $x$를 이용하고, $y_2$에 대한 확률을 구하기 위해 $x$와 $y_1$을 이용한다는 점에서 seq2seq은 conditional language model이다. 이를 수식으로 표현하면 아래와 같다. 

$$P(y|x) = P(y_1|x)P(y_2|x,y_1)...P(y_T|x,y_1,...,y_{T-1})$$

> 훈련 방법

![](https://i.imgur.com/Ni0jbq3.png)

연산을 통해 첫번째 $\hat{y_1}$ 예측값을 구하고 나면, cross entropy를 사용해 $J_1$을 구한다. 최종 loss function은 각 loss의 평균 값인 $J=\frac{1}{T}\sum_{t=1}^{T}{J_t}$으로 $J$로 back propagation을 진행한다. 

### 3) Decoding 방식

> 기존 방식: Greedy decoder

Seq2seq는 디코더의 각 단계에서 가장 높은 확률값을 갖는 단어 하나만 출력한다는 점에서 **greedy decoder**라 한다. 이 경우에는 하나의 출력값만 가지기 때문에 좋지 않은 번역 결과가 나왔을 때 뒤로 돌아갈 수 없다는 단점을 갖는다. 

> 개선 방법: Beam search decoder

디코더의 각 단계에서 높은 확률값을 갖는 k개의 단어를 출력하는 방법으로, k는 **beam size**라 한다. 이 때 score값은 greedy decoder를 사용할 때와 달리 log probability로 계산해 음수값을 갖는다.(log함수는 0과 1사이에서 음수 값을 갖는다)

![](https://i.imgur.com/sQAE39i.png)


beam size가 2라고 하면 각 단계마다 가장 높은 score를 갖는 단어를 두개를 선택해 계산을 진행한다. END 토큰이 나올 때까지 디코딩을 하며, 모든 hypothesis는 서로 다른 시점에서 END 토큰이 나온다. 한 문장에서 END 토큰이 나왔다면, 그 문장은 잠시 제외한 채 높은 Score를 갖는 다른 두개의 hypothesis로 beam search를 진행한다. 따라서 beam search는 **몇개의 단어가 나오면 학습을 중단할지 혹은 쌓아둘 문장의 수를 사전에 정하고** 그 기준치를 넘으면 학습을 중단한다.

hypothesis는 서로 다른 시점에서 END 토큰이 나오기 때문에 **짧은 문장일수록 좋은 score**를 가지게 됩니다. 따라서 loss function을 문장의 길이로 나눠주어 정규화해 사용한다.

$$\frac{1}{T}\sum_{t=1}^{T}{logP(y_i|x,y_1,y_2,...,y_{i-1})}$$

### 4) 성능 평가 : BLEU
    
**기계의 번역과 인간의 번역의 유사성**을 계산하는 방법으로 n-gram precision을 측정한다. 4-gram precision을 사용한다면 4개의 연속된 단어를 슬라이딩해 가면서 인간의 번역문장과 유사도를 계산하는 것이다. 하지만 이 평가 방식은 **짧은 문장일 수록 좋은 score**를 갖는다는 단점이 있다. 좋은 번역일수록 복잡한 문장구조와 상세한 정보들이 담겨야 하는데 이는 유사도를 측정하면 낮은 점수를 반환하기 쉽기 때문이다. 

### 5) 해결해야 하는 문제들
    
- 신조어를 반영하기 쉽지 않다.
- 학습데이터를 사전에서 가져오고 테스트 데이터가 구어체인 경우에 성능이 좋지 않다
- 책을 통째로 번역한다고 할 때 긴 문단 속에서 이전 문장들을 고려하기 위해서는 비용이 많이 든다. 
- 두 언어의 pair 형태로 데이터를 구축하는 것이 쉽지 않다 등


## 3. Attention

### 1) bottleneck problem

기존의 seq2seq 모델은 source sentence에 대한 모든 정보를 하나의 single vector로 표현하기 때문에 성능 저하가 발생한다 

### 2) Attention 구조
    
인코더의 모든 hidden state와 직접적인 연결을 하는 방식

> Attention scores

![](https://i.imgur.com/4lBFpIl.png)

decoder RNN의 Hidden state와 encoder RNN에서의 모든 Hidden state와의 벡터 내적을 통해 Attention scores를 구한다. 
t시점 디코더 RNN의 hidden state를 $S_t$ 인코더 RNN의 hidden state를 $h_1,h_2,...,h_N$라고 하자

$$score(s_t,h_i)=s_t^Th_i$$

attention score $e^t$는 다음과 같다

$$e^t = [s_t^Th_1,s_t^Th_2,...s_t^Th_N]$$

> Attention distribution($\alpha^t$)

![](https://i.imgur.com/ubP3Vs7.png)

Attention score에 softmax함수를 취해서 확률 분포를 구한다. 이 예시에서는 첫번째 단어인 il에 포커스를 두고 있음을 알 수 있다. 

$$\alpha^t=softmax(e^t)$$

> Attention output(**$a^t$**)

![](https://i.imgur.com/sx4jF7f.png)

각 attention 가중치와 인코더 hidden state의 가중합을 통해 Attention output을 구한다.

$$a_t=\sum_{i=1}^{N}{\alpha^th_i}$$

> ${\hat{y}}$ 구하기

Attention output **$a^t$** 와 디코더 hidden state $s_t$를 연결해준다 $v_t=[a^t;s_t]$

![](https://i.imgur.com/pIbghqg.png)

$$\tilde{s_t}=tanh(W_c[a^t;s_t]+b_c)$$ 

$$\hat{y}=Softmax(W_y\tilde{s_t}+b_y)$$

### 3) Attention 장점
    
- 병목현상을 해결
- attention distribution을 통해서 decoder가 어떤 단어에 focus를 두었는지 알 수 있다.
- vanishing gradient 문제 해결

[참고자료]
[딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/22893)