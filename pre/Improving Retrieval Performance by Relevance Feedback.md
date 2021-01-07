# Improving Retrieval Performance by Relevance Feedback
--- 
title: '[Information retrieval] Improving Retrieval Performance by Relevance Feedback '
usemathjax: true
use_math: true
comments: true
layout: single
classes : wide
categories:
  - Information retrieval
---
본 글은 문성빈 교수님의 <정보검색론> 수업시간에 과제로 진행한 [Improving Retrieval Performance by Relevance Feedback] 논문 요약 내용입니다.

## Introduction to Relevance Feedback

이용자들이 문헌집단에 대한 지식이 없는 환경에서 좋은 성능의 검색어를 구성하는 것은 어려운 일이다. 따라서 초기의 검색어는 적합성 피드백을 통해서 개선되어야 한다. 전통적으로 검색어 구성은 수동적인 과정이었으나, 1960년대 중반부터 자동으로 작성되기 시작했다. 적합성 피드백의 가장 핵심적인 목표는 검색된 적합 문헌에 등장한 중요한 용어의 가중치 값을 높여 질의를 재작성하는 것이다. 이는 검색어를 벡터상의 공간에서 적합 문헌에 가까워지도록 하여 해결이 가능하다.  

$$ Q_0 = (q_1,q_2,...,q_t)$$

이와 같은 벡터형태의 쿼리가 있다고 하자. $q_i$는 쿼리 $Q_0$에 포함된 용어$i$의 가중치를 의미하며, 0과 1사이의 값을 갖는다. 용어는 통제어 혹은 자연어의 형태다. 적합성 피드백 과정은 기존의 $Q_0$에서 새로운 벡터 $Q^{'}$를 생성하여 검색성능을 개선한다. 

$$Q^{'} = (q_1^{'},q_2^{'},...,q_t^{'})$$ 

새로운 용어는 기존에 0의 값을 갖던 가중치에 양수 값을 부여해 등장한다. 또한 부적합한 기존의 용어는 가중치 값을 0으로 만듦으로써 제거할 수 있다. 


## Basic Feedback Procedures

> Vector Processing Methods

문서를 $D=(d_1,d_2,...,d_t)$ 검색어를 $Q=(q_1,q_2,...,q_t)$ 벡터형태로 나타낸다고 하자. 문서와 검색의 유사도는 벡터 내적을 통해 구할 수 있다. 적합성 피드백을 통해 개선된 검색어 $Q_{opt}$를 구하기 위해서는 적합 문헌과의 유사도는 높이고 부적합 문헌과의 유사도는 최소화해야 할 것이다. 

$$Q_{opt} = argmax[sim(Q,적합문헌)-sim(Q,부적합문헌)] = \frac{1}{n}\sum_{적합문헌}\frac{D_i}{\left| D_i \right|}- \frac{1}{N-n}\sum_{부적합문헌}\frac{D_i}{\left| D_i \right|}$$ 

그 해답은 적합문헌 중심벡터와 부적합문헌의 중심벡터의 차이다.($\sum\frac{D_i}{\left| D_i \right|}$은 벡터의 centroid다 ) 하지만 실제 상황에서는 모든 문헌에 대해 적합성을 고려할 수 없기 때문에 아래와 같이 우리가 아는 범위 내에서 계산되어야 할 것이다. 

$$Q_{1} = Q_{0} + \frac{1}{n_1}\sum_{적합문헌}\frac{D_i}{\left| D_i \right|}- \frac{1}{n_2}\sum_{부적합문헌}\frac{D_i}{\left| D_i \right|}$$ 

여기서 $Q_{0}$은 초기 질의이며 $Q_{1}$은 한번의 적합성 검정이 완료된 질의를 나타낸다. 위 수식을 일반화 하면 다음과 같다.

$$Q_{i+1} = \alpha Q_{i} + \beta\sum_{적합문헌}\frac{D_i}{\left| D_i \right|}- \gamma\sum_{부적합문헌}\frac{D_i}{\left| D_i \right|}$$ 

![](https://i.imgur.com/nqsbTse.png)

이 그림을 살펴보면 $Q_{0}$ 검색어에 대해 $D_{1}$이 적합문헌인 경우에 적합성 검정 이후의 $Q^{'}$ 검색어는 벡터 공간 상에서 $D_{1}$에 인접해 있음을 확인할 수 있다. 

> Probabilistic Feedback Methods

확률기반의 피드백은 다음식의 결과를 내림차순으로 나열한 순위를 이용한다. 

$$log\frac{p(x|rel)}{p(x|nonrel)}$$

$p(x|rel),p(x|nonrel)$은 적합문헌 혹은 부적합문헌일 때 벡터 x를 가질 확률을 나타낸다. 
각 단어가 적합문헌과 부적합문헌에 독립이고, 문헌의 색인어 가중치가 binary한 경우에 문헌-검색어 유사도는 다음과 같이 표현될 수 있다

$$sim(D,Q)=\sum{d_ilog\frac{p_i(1-u_i)}{u_i(1-p_i)}} + constants$$

$$p_i = p(x_i=1|relevant), u_i=p(x_i=1|nonrelevant)$$ 

하지만 실생활에서는 모든 문서를 대상으로한 $p_i,u_i$를 계산할 수 없다. 따라서 초기검색에서는 주로 $p_i$에 상수를 부여하고(주로 0.5),$u_i$ 는 $n_i/N$로 사용한다. 
![](https://i.imgur.com/mUZhz0Y.png)


$$initial Sim(D,Q)=\sum{d_ilog\frac{N-n_i}{n_i}}$$

초기 검색 이후에 적합성에 대한 판단이 주어지면 

$$p_i = \frac{r}{R},\ u_i = \frac{n-r}{N-R}$$

값을 대입해 유사도를 구한다 

$$feedback\ sim(D,Q)=\sum{d_ilog\frac{r_i}{R-r_i}/\frac{n_i-r_i}{N-R-n_i+r_i}}$$

하지만 이 공식의 경우 R이나 r이 0에 가까운 작은 값을 가질 때 분모가 0이 되거나 log0이 되는 문제가 있다. 따라서 adjustment factor를 부여해 이 문제를 해결해야 한다.

- 분모 분자에 각각 상수 1과 0.5를 부여하는 경우

$$p_i = \frac{r+0.5}{R+1},\ u_i = \frac{n-r+0.5}{N-R+1}$$

- adjustment factor로 $n_i/N$을 사용하는 경우: 상수를 사용할 때보다 더 좋은 성능을 보인다고 한다.

$$p_i = \frac{r+n_i/N}{R+1}, u_i = \frac{n-r+n_i/N}{N-R+1}$$

## Relevance Feedback Evaluation 

이 실험에서는 relevance feedback 평가를 위해 6개의 문헌집단을 사용하였다. 초기검색에 대한 문헌과 쿼리의 유사도를 측정하기 위해 사용된 용어 가중치는 다음과 같다.

$$w_i=\frac{(0.5+0.5\frac{tf_i}{max tf})log\frac{N}{n_i}}{\sqrt{(0.5+0.5\frac{tf_i}{max tf})^2log\frac{N}{n_i}^2} }$$

실험에 있어 초기 검색의 상위 15개 문헌이 적합성 검정 및 feedback query를 구성하는데 사용되었다. relevance feedback의 성능을 평가하기 위해 recall과 precision을 기준으로 초기검색과 한번의 피드백이 수행된 검색 결과를 비교하였다. 하지만 feedback에 사용된 15개의 문헌은 한번의 iteration 이후에 상위 rank에 위치할 수 밖에 없다는 문제점이 존재했으며, 그 해결책으로 학습에 사용된 문헌들을 재검색에 사용하지 않는 residual collection system을 사용하였다.

> relevance feedback methods

6개의 문헌집단에 대해 12가지의 relevance feedback방법을 적용하였으며, 크게 vector modificaion 방법과 probabilistic feedback으로 구분할 수 있다. 

- Vector adjustment(Ide dec-hi): 적합문헌을 모두 포함하고, 가장 상위에 있는 부적합 문헌 하나를 제거하는 방식

$$Q_{new} = Q_{old} + \sum_{all적합문헌}{D_i}- \sum_{one부적합문헌}{D_i}$$

- Vector adjustment(Ide regular): 모든 적합/ 부적합 문헌을 제거하는 방식

$$Q_{new} = Q_{old} + \sum_{all적합문헌}{D_i}- \sum_{all부적합문헌}{D_i}$$

- Vector adjustment(standard rocchio): 용어 가중치를 적합문헌의 수와 부적합문헌의 수 $n_1,n_2$로 나눠주며, $\beta+\gamma = 1$이어야 한다. 

$$Q_{new} = Q_{old} + \beta\sum_{n_1적합문헌}\frac{D_i}{n_1}- \gamma\sum_{n_2부적합문헌}\frac{D_i}{n_2}$$

- probabilistic conventional: r과 R이 0이 되는 것을 방지하기 위해 분모 분자에 상수를 부여하는 방식

$$p_i = p(x_i=1|rel) = \frac{r+0.5}{R+1.0},\ u_i=p(x_i=1|nonrel)=\frac{n-r+0.5}{N-R+1.0}$$ 

- probabilistic adjusted derivation: r과 R이 0이 되는 것을 방지하기 위해 분모 분자에 1, n/N을 부여하는 방식

$$p_i = p(x_i=1|rel) = \frac{r+n/N}{R+1.0},\ u_i=p(x_i=1|nonrel)=\frac{n-r+n/N}{N-R+1.0}$$ 

- probabilistic adjusted derivation: probabilistic adjusted derivation 방법과 동일하지만 $r^{'}=r+3, R^{'}=R+3$을 사용하는 방식

위에서 언급된 feedback 방법들은 모두 쿼리에 대한 가중치를 업데이트하게 되지만, 문서에 대한 가중치는 구체화되지 않는다. 따라서 문헌 벡터는 tf와 idf로 구성된 $w_i$ 혹은 binary weight로 표현하였다.

> Query expansion

쿼리 확장 방법은 크게 두가지 방법이 있다. 쿼리를 확장하지 않고 기존의 쿼리에 대한 가중치 값을 조정하는 방식, 이전 검색으로부터 얻은 적합 문헌의 용어를 사용해 쿼리를 확장하는 방식이 있다. 쿼리 확장을 하는 경우에는 검색된 적합 문헌에서 가장 많이 등장한 용어를 주로 사용한다고 한다. 

> 평가 방법 

72개의 피드백 결과물에 대해 고정된 recall값 0.75 0.5 0.25에 대한  precision값의 평균값을 계산해 순위를 매겼다. Table4와 Table5는 5개의 문헌집합에 대해 문서 가중치로 각각 $w_i$, binary weight를 사용했을 때의 결과물이며, 각 문헌에 대한 최상위/최하위 순위를 기록한 방법들이 일관되었음을 확인할 수 있다.  

![](https://i.imgur.com/5raWB6n.png)

![](https://i.imgur.com/jJ9dgM8.png)

> Table 4-5 평가 결과

- 가중치를 부여한 문헌벡터가 binary weight를 사용할 때보다 성능이 좋다 
- common terms로 제한된 확장보다 전체 문헌을 사용하였을 때 성능이 더 좋았다 
- 가장 좋은 성능을 보인 방법은 Ide dec-hi였다.
- 로치오 방식을 사용할 때는 $\beta>\gamma$일 때 더 좋은 성능을 보였다. 
- 일반적으로 확률 기반의 피드백은 vector modificaion보다 성능이 떨어지지만 컴퓨팅 기반 환경에서 더 선호되는 방법이다. 
- 초기 검색어의 길이가 짧을수록 성능이 향상되었다
- 초기 검색 결과가 나쁠수록 성능이 향상되었다
- 문헌집단이 기술적일수록 성능이 더 향상되었다. 

> Table 6 평가 결과

![](https://i.imgur.com/5XF1aHk.png)

Table 6는 NPL 문헌집단에 대한 결과물로 5개의 문헌집단과 다른 패턴의 결과가 도출되었다. NPL 문헞집단은 다른 문헌집단에 비해 문헌과 검색어 벡터의 길이가 매우 짧으며, TF값이 평균 1.21로 매우 낮다. 따라서 TF 가중치나 길이 정규화 작업이 무의미하며, Binary weight가 선호된다. 따라서 검색어로 자연어보다는 통제어를 사용할 때 더 좋은 성능을 보인다. 





