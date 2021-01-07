--- 
title: '[NLP]Lecture3'
use_math: true
comments: true
layout: single
categories:
  -  NLP

---

본 글은 스탠포드 [CS 224N](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/) [Word Window Classification, Neural Networks, and Matrix Calculus] 강의를 들으며 정리한 글 입니다.


## 1. Classification
NLP에서 분류는 단어, 문자, 문서 등을 x값으로 받아 감정, named entities 등 y값을 예측하는데 사용된다. binary case인 경우에는 시그모이드 함수를 사용하며, class가 3개 이상인 경우에는 **소프트맥스** 함수를 이용해 y값을 예측한다

> Softmax

소프트맥수 함수는 $pi =\frac{e^{z_i}}{\sum_{j=1}^{k}{e^{z_j}}}   $ 형태이며, 인풋값을 넣으면 0과 1 사이의 확률값으로 정규화해준다. 소프트맥스 함수에서 $e$를 사용하는 이유는 자연상수 $e$는 큰 값을 더 크고 작은 값은 더 작게 만들어주는 성질이 있기 때문이다. 분류에서 우리의 목적은 실제(y) class c에 대한 예측 probaility를 maximize 하는 것으로, $e$를 사용하면 실제 class에 대한 확률을 maximize할 수 있다. 이제 예시를 들어보도록 하겠다. class가 총 3개가 있다고 하면, output은 각 class에 대한 값으로 3차원의 형태일 것이다. 3차원 벡터 $z=[z_1,z_2,z_3]$의 입력을 받은 소프트맥수 함수의 출력값은 아래와 같다 

![softmax](http://whdbfla6.github.io/assets/images/nlp3-1.JPG)

$p_1,p_2,p_3$ 각각의 값은 1번 클래스가 정답일 확률, 2번 클래스가 정답일 확률, 3번 클래스가 정답일 확률을 나타내며 총 합은 1이다. 

![softmax](http://whdbfla6.github.io/assets/images/nlp3-2.JPG)

이 데이터가 실제로 2번 클래스에 속한다면, one-hot vector는 $[0,1,0]$다.이제 실제값과 예측값의 오차를 계산해 비용함수를 구해야 하는데, 일반적으로 Cross Entropy를 사용한다.

> Cross Entropy 

Cross Entropy에 대해 설명하기에 앞서 Entropy의 개념을 살펴보자. Entropy는 불확실성의 척도로 다음과 같은 수식으로 표현된다. 

$H(q) = - \sum_{c=1}^{C}{q(c)*log  q(c)}$ 


$H(q)$ 는 불확실성이 커질수록 큰 값을 갖는다. 예를들어 가방 안에 빨간 공과 파란 공이 있다고 하자. 이 때 두 공의 개수가 동일한 경우가 빨간 공과 파란 공의 비율이 1:9 인 경우보다 Entropy가 더 클 것이다. 공의 비율이 1:9인 경우는 파란공이 뽑힐 가능성이 높다는 확실성을 갖지만, 비율이 같으면 어떤 공이 나올지 모르기 때문이다. 수식에 넣어보면, $-(0.1*log(0.1)+0.9*log(0.9))=0.14 ,-(0.5*log(0.5)+0.5*log(0.5))=0.30 $ 으로 공의 개수가 동일한 경우 불확실도가 높은 것을 확인할 수 있다. Entropy를 기반으로 하여 나온 loss function이 Cross Entropy다. 수식의 형태는 아래와 같다. 

$H(p,q) = - \sum_{c=1}^{C}{p(c)*log  q(c)}$ 

($p$는 실제 확률 분포, $q$는 계산된 확률)

직관적으로 이해해보면 실제 y값은 $[0,1,0]$이고, 예측된 y값이 정확한 경우에 $q(c)$는 1이어야 한다. 이를 식에 대입해보면 $-1log(1) = 0$ 즉 Cross Entropy의 값이 0이 된다. 불확실성을 낮추기 위해서는 $- \sum_{c=1}^{C}{p(c)*log  q(c)}$ 의 값을 최소화하는 방향으로 학습해야 한다. 

Cross Entropy loss function을 전체 데이터셋에 적용한 최종적인 비용 함수 형태는 다음과 같다.

$J(theta) =  \frac{1}{N}\sum_{i=1}^{N}{-log(\frac{e^{f_y}}{\sum_{c=1}^{C}{e^{f_c}}})}$

## 2. NN Classifiers

![NN](http://whdbfla6.github.io/assets/images/nlp3-3.JPG)

Neural network classifier는 nonlinear한 decision boundary를 구현하기 위해 등장했다. 앞서 배운 소프트맥스 함수의 decision boundary는 linear하기 때문에 매우 제한적이다. 왼쪽 그림을 살펴보면 빨간색 영역에 초록색 점들이 많은 것을 확인할 수 있는데, 이 문제를 해결하기 위해서는 직선이 아닌 곡선 형태의 boundary가 필요하다. 이 문제를 뉴럴 네트워크가 해결해준다. 

![NN](http://whdbfla6.github.io/assets/images/nlp3-4.JPG)

뉴럴 네트워크의 형태는 다음과 같다. input layer와 output layer의 사이에 은닉층이 존재하며, 이는 nonlinear한 decision boundary를 구현할 수 있도록 한다. 더 복잡한 문제를 해결하기 위해서는 수많은 은닉층을 추가하면 된다. 이 때 은닉층이 하나인 경우는 다층 퍼셉트론(MLP), 은닉층이 2개 이상인 경우 심층 신경망(DNN)이라 한다.

![NN](http://whdbfla6.github.io/assets/images/nlp3-5.JPG)

은닉층을 통과하고 나오면 최종 output을 도출하기 이전에 활성화 함수를 통과해야 하는데, 활성화 함수는 선형함수가 아닌 비선형 함수여야 한다. 선형함수를 사용하게 되면 hidden layer를 계속 쌓더라도 하나의 layer를 통과한 것과 차이가 없어진다. 예를들어 $f_1(x)=w_1x, f_2(x)=w_1x $라고 하면 $f_1(f_2(x))=w_1w_2x$로 선형함수 형태다. 즉 선형함수를 무수히 많이 통과하더라고 최종 결과는 하나의 선형 함수로 표현이 가능하며, 신경망이 깊어지더라도 흥미로운 계산을 할 수 없음을 의미한다.  

> NLP deep learning

![NN](http://whdbfla6.github.io/assets/images/nlp3-6.JPG)

NLP에서는 파라미터(w)와 word vector(x)를 같이 학습시킨다는 점에서 일반 딥러닝 학습과 차이가 있다. 이 때 훈련데이터가 매우 적은 상황에서 pre-trained word vector를 임베딩 벡터로 사용하기도 한다. 훈련데이터가 적은 경우에는 해당 문제에 특화된 임베딩 벡터를 만드는게 쉽지 않다고 한다. 따라서 해당 문제에 특화된 것은 아니지만 일반적이고 많은 훈련데이터로 이미 학습된 임베딩 벡터를 사용하면 성능을 개선할 수 있다.

## 3. Window Classification

Window classification의 경우 NER(Named Entity Recognition)을 예시로 들어 설명을 할 것이다. NER은 문장 안에서 Location, Person, Organization 등 개체명(Named Entity)를 분류하는 방법론이다.‘디카프리오가 나온 영화 틀어줘’라는 문장에서 ‘디카프리오’를 사람으로 인식하는 것을 목표로 한다. 하지만 NER의 경우,<br/>
1. Entity 경계를 정하기 힘들다
(First National Bank vs National Bank 어느것을 input data로 볼 것인가?) <br/>
2. Entity 구분하기 힘들다(Future school 미래의 학교를 의미하는지, 그 자체가 학교명인지 구분이 어려움)<br/>
3. Unknown entity에 대한 class를 얻기 힘들다

는 문제점을 갖는다. 따라서 토큰화와 품사 태깅 전처리를 끝내고 난 상태를 입력으로 한다. 전처리를 거쳐 word vector를 구하고 나면 window를 얻을 수 있으며, input의 형태는 크게 두가지 방법으로 구현할 수 있다. 

> 방법1: Window에서 word-vector의 평균 구하기

![NER](http://whdbfla6.github.io/assets/images/nlp3-7.JPG)

첫번째 방법은 각각의 word vector의 평균값을 구하는 것이다. 위 그림은 window size가 2인 경우로 5개의 4차원 vector를 갖는다. 단순히 평균을 구하게 되면 하나의 4차원 벡터로 압축이 되는데 이는 위치 정보를 다 삭제하는 결과를 초래한다.그래서 일반적으로 방법 1이 아닌 방법 2를 사용해 input으로 집어넣는다. 

> 방법2: 모든 word-vector concatenate하기

![NER](http://whdbfla6.github.io/assets/images/nlp3-8.JPG)

두번째 방법은 모든 word vector를 하나로 잇는 것이다. window에는 5개의 4차원 vector가 있기 때문에 input은 20차원의 벡터가 된다. 이는 위치 정보를 보존할 수 있는 장점이 있다. concatenate된 벡터를 인풋으로 넣게 되면 hidden layer를 통과하게 되는데 이 때 활성화 함수는 비선형 함수를 사용해야 한다. 비선형 함수를 사용하게 되면 각 단어 간의 비선형적 관계를 나타낼 수 있기 때문이다. 예를들어 첫번째 단어가 museum이고 두번째 단어가 in이나 or 같은 전치사인 경우에 그 다음 center 단어가 location일 수 있다는 좋은 신호가 된다. 즉 활성화 함수가 비선형 함수 일 때 단어들의 interaction term을 잘 설명할 수 있게 된다. 최종 output s의 경우 확률 값으로 변환하거나, 그대로 사용하기도 한다. 


**Case1 : 확률 값으로 변환**

이는 output s를 소프트맥스 함수를 이용해 확률값으로 변환해주는 방법으로 앞서 자세히 설명했기 때문에 생략하도록 하겠다.

**Case2 : 그대로 사용(Unnormalized scores)**

이는 가운데 단어가 Location인지 아닌지 분류하는 binary classification 문제를 해결하는데 사용된다. 

![NER](http://whdbfla6.github.io/assets/images/nlp3-9.JPG)

여기에서 가운데 단어 paris가 location 인지 아닌지 확인하고 싶다고 하자. Corpus 내 모든 위치에서 학습시켜야 하며, 이는 negative sampling 적용하는 것과 같다. input은 True window와 Corrupt window 두 종류를 사용한다. True window는 *museums in paris are amazing* 과 같이 center 단어가 location인 경우이며, Corrupt window은 center 단어가 NER location으로 label되지 않은 경우로 *Not all museums in paris*가 해당 예다. input vector를 layer에 통과시키면 최종적인 s값이 나오게 되는데, loss function으로는 Min-margin loss를 사용한다.Min-margin loss는 $max(1+s_c-s,0)$로 이 값을 최소화하는 방식으로 훈련을 진행해야 한다. 핵심 아이디어는 true window의 score는 크게 corrupt window의 score는 낮게 나오도록 하는 것이다.($s$는 true window's score, $s_c$는 corrupt window's score) $s-s_c>0$인 경우 true window의 score가 더 크기 때문에 학습을 중단해도 되지만, $s_c-s>0$인 경우 학습이 진행되어야 한다. loss function에서 1은 margin으로 true window와 corrupt window의 구분을 명확하게 해준다.



















참고문헌<br/>

[딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/35476)<br/>
[DSBA](http://dsba.korea.ac.kr/seminar/?category1=Lecture%20Review&mod=document&pageid=1&uid=42)
