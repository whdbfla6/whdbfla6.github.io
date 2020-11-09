---
title: "[Information retrieval]Term-Weighting Approaches In Automatic Text Retrieval 논문 요약"
use_math: true
comments: true
layout: single
classes : wide
categories:
  - Information retrieval

---

## 1. Automatic Text Analysis
>Automatic text retrieval system

자동 텍스트 검색 시스템은 저장된 텍스트와 이용자의 quary에 포함된 content identifier(내용 식별자)의 비교를 기반으로 구성된다. query와 document 간의 유사도가 높을 수록 해당 문서가 상단에 검색될 수 있도록 하는 것이다. *Content identifier*는 문서의 텍스트나 quary에서 추출해 사용한다. 문서나 쿼리는 다음과 같은 형태로 표현할 수 있다.


$$D=(t_0,w_{d_0};t_1,w_{d_1};,...,;t_t,w_{d_t})$$ 

$$Q=(q_0,w_{q_0};q_1,w_{q_1};,...,;q_t,w_{q_t})$$ 

여기서 $t,q$는 term vectors에 해당하고, $w_k$는 문서D에서 term $t_k$의 가중치를 나타낸다. 이제 query와 document의 유사도를 측정해야 하는데 이는 가중치($w$)로 이루어진 t차원 벡터의 내적으로 계산된다. 

$$D=\begin{bmatrix}w_{d_0}\\w_{d_1} \\ \vdots\\w_{d_t}\end{bmatrix}\quad Q=\begin{bmatrix}w_{q_0}\\w_{q_1} \\ \vdots\\w_{q_t}\end{bmatrix}\quad$$ 

$$similarity(Q,D)=\sum_{k=1}^t{w_{q_k}\cdot w_{d_k}}$$ 

가중치는 해당 문서에 등장하는 경우 1 아닌 경우 0으로 binary한 값을 사용할 수도 있지만, 중요한 단어의 경우 1에 가깝고 중요하지 않은 단어는 0에 가깝도록 0과 1사이의 연속적인 값을 사용할 수도 있다. 상황에 따라 정규화한 weight를 사용하기도 하는데, 이는 가중치 값을 벡터의 크기로 나눠준 값이다. 정규화 과정을 통해 가중치가 0과 1사이의 값으로 반환된다. 

$$ D=\begin{bmatrix}w_{d_0}\\w_{d_1} \\ \vdots\\w_{d_t} \end{bmatrix}\quad (noramlized\ weight) = \frac{w_{dk}}{\sqrt{\sum_{i=1}^t{(w_{d_i})^2}}}\cdots(a) $$ 

(a)수식을 기존의 유사도 공식에 대입을 하면 다음과 같다.

$$similarity(Q,D)=\frac{\sum_{k=1}^t{w_{q_k}\cdot w_{d_k}}}{\sqrt{\sum_{k=1}^t{(w_{q_k})^2}\cdot\sum_{k=1}^t{(w_{d_k})^2}}}$$

$t_1$: 바나나 $t_2$: 딸기 $t_3$: 원숭이 $t_4$: 사과  $t_5$: 좋다 $t_6$: 아침 <br/>

문서1: **사과**는 맛있다 
문서2: **원숭이**는 **바나나**를 **좋아한다** 
문서3: **딸기**와 **사과** 
문서4: **아침**에 **사과**를 먹는 것이 **좋다** <br/>

query: **원숭이**가 과일을 **좋아하나**?<br/>

위와 같은 예시를 binary한 형태로 가중치를 부여한다면, 문서와 query를 다음과 같은 행렬로 표현할 수 있다. 

![](https://i.imgur.com/J5zDsZj.png)


정규화된 유사도 공식에 대입을 하면,
$similarity(Q,D_1)=0\quad similarity(Q,D_2)=\frac{2}{\sqrt{6}}\quad similarity(Q,D_3)=0$  $similarity(Q,D_4)=\frac{1}{\sqrt{6}}$ 이다. 
유사도 값에 따라 D2, D4의 문서가 차례로 검색될 것이다.<br/>

automatic text retrieval system을 구축하는데 있어 크게 고려해야 하는 사항은 두가지이다. 
1. 문서와 query를 표현하기 위해서 어떤 content term을 사용할 것인가? 
2. 가중치의 값이 중요한 용어가 무엇인지 구분지어 줄 수 있는가?

> content term 결정

많은 경우에 content-term으로 텍스트나 query에서 추출된 single-term을 사용했을 때 좋은 검색 성능을 보였다. 하지만 single term이 문서의 내용을 완벽하게 구분지어줄 수 없기 때문에 복잡한 text representation을 위해 연구가 이루어지고 있다. 복잡한 text representation 방법의 예는 다음과 같다.  

1. related term: 문서에서 동시에 단어들이 출현하는 경우에 서로 연관성이 높다는 전제에서 시작되며, 동시 출현 행렬에서 related term을 생성하는 방법이다.
2. term phrases : 단어의 등장 횟수나 다른 통계적인 기법을 기반으로 term phrases를 생성하는 방법이다. 이는 의존적인 term 그룹을 미리 추출하지 않은 경우, 새로운 문서에 적용할 때 좋지 못한 성능을 보인다.
3. 단어 그룹화: 상위어 밑에 동의어 세트를 여러개 구성하는 방법으로 상위어를 내용을 구분지어 주는 content identification으로 사용하는 방법이다. 이 방법의 경우 사람이 직접 업데이트를 해야하기 때문에 구축 비용이 많이 든다. 
4. 지식을 기반으로 추출된 항목을 문서와 query를 표현하는데 사용하는 방법으로 '단어의 그룹화'와 같은 이유로 선호되지 않는다.

엄격한 기준으로 복잡한 text representation을 하는 경우에 사용 가능한 식별자는 많지 않고, 검색 성능이 single-term을 사용할 때와 큰 차이가 없어진다. 또한 완화된 기준을 적용해 많은 식별자를 얻을 수 있더라도 성능이 좋지 않다고 한다. 결과적으로 검색 식별자로 single-term이 선호되며, 각 용어를 구분하기 위해 가중치 factor를 사용하게 된다. 

## 2. Term-Weight Specification
>Recall and Precision

Term-weighting system의 가장 중요한 기능은 검색 성능을 높여준다는 것이다. 효과적인 검색 성능을 나타내는 두가지 척도는 재현율과 정확률이다. 

$$recall = \frac{검색된\ 적합문헌\ 수}{적합문헌\ 총\ 수}$$ 

$$precision = \frac{검색된\ 적합문헌\ 수}{검색된\ 총\ 문헌\ 수}$$ 

재현율은 검색 결과에 이용자가 원하는 내용을 얼마나 포함해 주는지에 따라 결정되며, 정확률은 이용자가 원하지 않는 내용이 검색 결과에 나오지 않을 수록 높은 값을 갖는다. 재현율은 문헌 collection에 자주 등장하는 단어를 검색어로 사용하면 관련 문헌이 많이 검색되면서 성능을 높일 수 있다. 반면에 정확률은 관련 없는 문헌들을 배제하기 위해 구체적이고 한정된 단어를 사용해야 한다. 즉 재현율과 정확율을 동시에 높이는 것은 거의 불가능한 일이다. 일반적으로는 포괄적인 단어를 사용해 합당한 재현율을 유지하면서 낮지 않은 정확률을 보장하는 것을 목표로 한다. 

>TF IDF

재현율과 정확률을 모두 보장하기 위해 term freqeuncy(TF)와 inverse document frequency(IDF) 두가지를 가중치 요소로 사용한다. **TF**는 해당 단어가 문헌이나 Query에서 몇 번 등장했는지를 측정하는 것으로, 재현율을 높이는 척도로 사용된다. **IDF**는 $log{N/n}$(N은 collection의 문헌 수, n은 해당 단어가 등장한 문헌 수)로 계산되며, 단어가 많은 문헌에서 등장하지 않을 수록 높은 값을 갖는다. 예를들어 해당 단어가 높은 TF 낮은 IDF값을 갖는 경우, 이 단어는 모든 문헌에서 등장하는 중요하지 않은 단어임을 알 수 있다. 즉 IDF는 정확률을 높여주는 척도로 사용된다. 따라서 단어의 중요도는 TF IDF 공식을 통해 계산된다. 하지만 단순히 TF IDF를 곱한 값을 TERM-weighting 요소로 사용하게 되면, 긴 문헌이 검색될 가능성이 높아지는 문제점이 생긴다. 따라서 기존 TF IDF를 정규화해 문헌 vector의 길이를 동일시한다면 이 문제를 해결할 수 있다. 

$$normalized\ term\ weighting = \frac{w}{\sqrt{\sum_{vectori}{(w_i)^2}}}$$

## 3. Term-Weighting Experiments
![](https://i.imgur.com/ZfZQ12x.png)

>term-weighting components

왼쪽 표는 term-weighting을 구성하기 위한 TF, IDF, 정규화요소의 조합을 보여주고 있다. TF 요소에서 b는 해당 단어가 등장하면 1 아니면 0으로 binary wieght, t는 일반적인 TF 이며, n은 정규화된 TF로 0.5와 1 사이의 값을 갖는다. IDF에서 x는 역문헌빈도를 고려하지 않으며, f는 일반적인 IDF, p는 확률적인 역문헌 빈도를 나타낸다. 정규화 요소는 정규화하지 않는 x와 벡터의 크기로 나눠주는 c 두가지가 있다. 

> typical term-weighting formulas 

오른쪽 표는 유명한 term-weighting 시스템을 나타내고 있다. coordination level은 문서와 query에서 동시에 등장하는 단어의 수를 반영하는 방법이다. Binary term independence는 coordination level에서 더 나아가 query에서 확률적인 역문헌빈도를 가중치로 사용하고 있다. Best fully weighted system에서는 문서에 정규화된 TF IDF를 query에는 정규화하지 않은 TF IDF를 사용한다.  

> 실험 대상 문헌

저자는 위의 weighting 공식 중 어떤 가중치 조합이 가장 좋은 성능을 나타내는지 실험하기 위해 6개의 문헌 집단을 사용했다(CACM CISI CRAN INSPEC MED NPL). 가장 작은 문헌 집한은 MED collection으로 1033개의 문헌과 30개의 query를 포함하고 있다. 이에 반해 가장 큰 문헌집단은 INSPEC으로 12684개의 문헌과 84개의 query를 포함하고 있다. NPL문헌집단의 경우 자연 언어 형태가 아닌 벡터로 인덱싱되어 있다는 점에서 다른 문헌과 구분되어 진다. 또한 각각의 query term이 하나의 query에서만 등장하며, 문헌에서의 평균 TF가 1.21이라는 점에서 다른 문헌에 비해 평균적인 TF가 현저히 낮다. 이 경우에는 정규화를 하는 것이 의미가 없어진다. 

> 실험 결과

저자는 재현율 값이 각각 0.25 0.5 0.75일 때 정확률을 측정하여 성능을 평가했다. 중복을 제외하고 287개의 가중치 부여 방식이 존재했으며, 가장 성능이 좋은 것을 1 나쁜 것을 287 rank로 부여했다. 6개의 문헌에 대해 대표적인 8개 방식에 대해 rank를 부여했는데 결과는 다음과 같다. 
![](https://i.imgur.com/Vj8VERz.png)

특수성을 가진 NPL 문헌집단을 제외하고 전반적으로 동일한 결과가 나왔다. 결과를 살펴보면 문헌은 정규화 term을 넣을 때 query는 nfx 방식을 사용할 때 가장 좋은 성능을 보였다. NPL 문헌의 경우 binary query weight와 정규화하지 않은 문헌 벡터를 사용할 때 가장 좋은 성능을 보였다. 실험 결과를 바탕으로 결과를 정리하면 다음과 같다.

1. Query vetors
- TF: 짧은 query 벡터의 경우 각각의 term이 중요하기 때문에 n을 사용한다. 긴 query 벡터의 경우 단어 등장 빈도에 따른 큰 구별이 지어져야하기 때문에 t를 사용하며, 등장 빈도가 모두 1로 동일한 경우에 TF term을 무시해도 된다.
- IDF: 거의 동일한 결과이나 f가 선호된다
- 정규화: query에서는 정규화가 큰 영향이 없기 때문에 x를 사용한다
2. Document vectors
- TF: 전문성 있는 단어(의학 용어)를 사용할 때 n을 사용하며, 다양한 단어를 사용할 때는 t를 사용하는 것이 좋다. 통제어를 기반으로 한 짧은 문헌의 경우 b=1을 사용한다.
- IDF: f와 p가 거의 동일한 성능을 보이나 일반적으로 f를 사용한다
- 정규화: 벡터 길이의 편차가 큰 경우에 c를 사용하며 동일한 길이의 짧은 문헌의 경우 정규화를 하지 않아도 된다. 







