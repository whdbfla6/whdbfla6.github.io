--- 
title: '[선형대수]Least Squares: Four ways'
use_math: true
comments: true
layout: single
classes: wide
categories:
  - 딥러닝기초
  - 선형대수
tags:
  - 딥러닝기초
  - 선형대수
---


## Least Squares: Four ways

$Ax=b$의 exact solution이 없는 경우에 근사해 $\widehat{\boldsymbol{x}}$를 찾는 방법 4가지가 있다.

1. pseudoinverse $A^+$를 통해 $\widehat{x}=A^{+} b$ 구하기
2. $A$가 n개의 독립적인 column을 가질 때 $A^{\mathrm{T}} A \widehat{x}=A^{\mathrm{T}} b$
3. gram-schmidt를 통해 $A=Q R$ QR분해하는 방법
4. $\|b-A x\|^{2}+\delta^{2}\|x\|^{2}$ penalty term 부여하는 방식

### 방법1 :  Pseudo inverse 이용

- Pseudo inverse는 $AA^{+} \sim I$를 만족하는 행렬. 즉 역행렬이 존재하지 않을 때의 근사 행렬
- $A$가 mxn 사이즈 인 경우에 $A^{+}$는 nxm사이즈다

![](http://whdbfla6.github.io/assets/linear-algebra/img6.PNG)

- 행렬 A에 row space vector x를 곱하는 경우 Ax는 column space에 있다. 
- 즉 x가 row space에 있는 경우 $A^{+}Ax=x$ b가 column space에 있는 경우  $A^{+}Ab=b$
- null space of $A^{+} =$ nullspace of $A^T$

> Pseudo inverse 

$$
A=U\boldsymbol{\Sigma} V^T \ A^{+}=V\boldsymbol{\Sigma}^{+}  U^T
$$

![img5](http://whdbfla6.github.io/assets/linear-algebra/img5.PNG)

- A의 역행렬이 존재하는 경우 $A^{-1}=V\boldsymbol{\Sigma}^{-1}  U^T$
- A의 역행렬이 없는 경우 일부 특이값은 0 

$$
Ax=b,\  x^+=A^{+}b
$$

![img7](http://whdbfla6.github.io/assets/linear-algebra/img7.PNG)

$x^{+}=A^{+}b$는 least square solution이며, least square solution을 만족하는 해들 중에서 norm이 가장 작다

A가 full rank로 단일 해 $x^{+}$가 존재한다고 하자. nullspace에 있는 벡터 x는 아무런 영향이 없기 때문에 $x^{+} +x$ 또한 해가 된다. $x^{+}$는 rowspace vector, $x$는 nullspace vector이기 때문에 $\parallel x^{+} + x \parallel = \parallel x^{+} \parallel + \parallel x \parallel$ 를 만족한다.  여기서 $x^{+}$을 해로 고를 때 norm이 가장 작아지는 것을 확인할 수 있다.

![img8](http://whdbfla6.github.io/assets/linear-algebra/img8.PNG)

### 방법2 : Projection 

> Ax=b 해가 존재하지 않는 경우

![img3](http://whdbfla6.github.io/assets/linear-algebra/img3.PNG)

- $Ax$는 행렬 $A$ column vector의 선형결합이다. 따라서 b가 A의 column space에 존재하지 않는 경우 해가 없다
- 벡터 b를 A column space에 가장 가까운 벡터로 바꾼 후에 근사해 $\widehat{x}$를 구하자
- error를 가장 최소화하기 위해서는 벡터 b를 column space에 정사영해야 한다. 

> 2차원 벡터 투영

- error vector와 $p$의 orthogonal한 관계를 이용해 $p$를 구하자

$$
a^T\cdot (b-xa)=0 \\ x=\frac{a^Tb}{a^Ta}\\ p=ax=a\frac{a^Tb}{a^Ta}=\frac{aa^T}{a^Ta}b
$$



> n차원 벡터 투영

$$
A=\left[\begin{array}{ll}
\mid & \mid \\
\boldsymbol{a}_{1} & \boldsymbol{a}_{2} \\
\mid & \mid
\end{array}\right] \ A\widehat{x}=p\ 라하자\\\\ a_1^T(b-A\widehat{x})=0, a_2^T(b-A\widehat{x})=0\\\\
\left[\begin{array}{c}
\boldsymbol{a}_{1}^{T} \\
\boldsymbol{a}_{2}^{T}
\end{array}\right](\boldsymbol{b}-A \hat{x})=\left[\begin{array}{l}
0 \\
0
\end{array}\right]\\\\ A^{T}(\boldsymbol{b}-A \hat{x})=0,\ \hat{x}=(A^TA)^{-1}Ab
$$

> 투영행렬이란?(=P) 

- 정사영 벡터를 구하기 위한 일반화된 matrix로 임의의 벡터에 적용 가능하다

$$
P=A(A^TA)^{-1}A
$$

- $Pb=p$ 투영행렬의 column space는 a가 지나가는 선. 즉 b에 투영행렬을 곱해서 a가 지나가는 선에 안착시키는 것
- $P^T=P\ P^2=P$

> $A^TA$는  언제 역행렬이 존재하는가

- $A$의 column이 independent할 때 역행렬이 존재한다. (full rank)
- If $Ax=0$ then $x=0$
- 왜?  $A$와 $A^TA$의 null space가 동일해서

### 방법3 : Gram-schmidt 

- $(A^TA)^{-1}\hat{x}=A^Tb$ 식을 통해 해를 구할 때 A의 column들이 orthogonal하다면 $(A^TA)$를 구하는 연산과정이 단순해진다
- 방법2를 이용할 때보다 연산과정은 두배지만 수치 안정성에 기여
- 그람슈미트 과정을 통해 행렬 A의 column들을 orthogonalize하자

> Gram-schmidt

- projection을 이용해서 구할 수 있다
- $R^n$ 의 subspace $w$ basis가 {$a_1, \cdot , a_n$}라 하자

$$
q_1 = a_1/ \parallel a_1 \parallel\\ 
A_2 = a_2 -(a_2^Tq_1)q_1,\ q_2 = A_2/ \parallel A_2 \parallel\\ 
A_3 = a_3 -(a_3^Tq_1)q_1 -(a_3^Tq_2)q_2,\ q_3 = A_3/ \parallel A_3 \parallel\\
$$

- 각 q 벡터의 크기는 1이고, 서로 다른 q벡터의 내적은 0이다
- 그람슈미트를 통해 구한 {$q_1, \cdot , q_n$}를 orthonormal basis라 한다

> QR분해

![](http://whdbfla6.github.io/assets/linear-algebra/img9.PNG)

- Q는 orthonormal basis로 구성된 행렬
- $R = Q^TA$ $r_ij = q_i^Ta_j$로 상삼각행렬이다

> gram schmidt with column pivoting

- 그람슈미트에서 $a_1,\ a_2  \cdot , a_n$ 순서 그대로 직교벡터를 만드는 것은 위험하다(행렬 elimination 과정에서 row exchange를 안 하는 것과 유사한 이유)
- 매 스탭마다 열 교환을 통해 가장 큰 길이의 벡터를 선택해야 한다

![](http://whdbfla6.github.io/assets/linear-algebra/img10.PNG)

- good column comes first!

### 방법4 : Penalty 부여

- A 행렬의 열이 dependent한 경우 nullspace가 nontrivial하며 $A^TA$ 또한 역행렬이 존재하지 않는다. 즉 방법2와 방법3을 사용할 수 없다
- 이 때 제안된 방법이 ridge regression!

![](http://whdbfla6.github.io/assets/linear-algebra/img11.PNG)

- penalty $\delta$가 0인 경우 pseudo inverse로 계산하는 것과 동일

$$
A=\sigma\\ x=\frac{\sigma}{\sigma ^2+\delta ^2},\ \delta \rightarrow 0\\
x=0(if\ \sigma=0), x=\frac{1}{\sigma}(if\ \sigma \neq 0)\\
$$

- 결과가  $\boldsymbol{\Sigma}$를 구하는 과정과 동일
- penalty term은 singular 값에 0이 존재해 역행렬이 없거나, near singular case인 경우 적용하는 것이 좋다. 여기서 특이값이 0에 가까운 아주 작은 값일 경우 near singular case라 하고 역행렬이 지나치게 커지는 문제를 초래한다. 이는 분모에 penalty term $\delta$를 부여해 해결이 가능하다

![img12](http://whdbfla6.github.io/assets/linear-algebra/img12.PNG)

> Ridge vs Lasso

Ridge는 penalty term으로 L2 norm을 사용하고, Lasso는 L1 norm을 사용한다. 
$$
\|\mathbf{x}\|_{p}:=\left(\sum_{i=1}^{n}\left|x_{i}\right|^{p}\right)^{1 / p}\\ \|x\|_{1}=\sum_{i=1}^{n}|x_{i}|,\ \|\boldsymbol{x}\|_{2}=\sqrt{x_{1}^{2}+\cdots+x_{n}^{2}}
$$

- 회귀모델에 적용하는 경우 ridge는 파라미터를 0에 가깝게는 하지만 완전히 0이 되지는 않는다. 설명변수의 영향력을 감소시키는 정도
- lasso는 일부 파라미터를 정확히 0으로 만들어서 변수선택에 사용됨
