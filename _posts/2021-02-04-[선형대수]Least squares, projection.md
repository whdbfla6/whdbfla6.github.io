---
title: '[선형대수]Least Squares: Projection'
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

## 1. Ax=b 해가 존재하지 않는 경우

![img3](http://whdbfla6.github.io/assets/linear-algebra/img3.PNG)

- $Ax$는 행렬 $A$ column vector의 선형결합이다. 따라서 b가 A의 column space에 존재하지 않는 경우 해가 없다
- 벡터 b를 A column space에 가장 가까운 벡터로 바꾼 후에 근사해 $\widehat{x}$를 구하자
- error를 가장 최소화하기 위해서는 벡터 b를 column space에 정사영해야 한다. 



## 2. 벡터 투영

> 2차원 벡터 투영

![](http://whdbfla6.github.io/assets/linear-algebra/img17.png)

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
\end{array}\right] \ A\widehat{x}=p\ 라하자
$$

$b-p=b-A\hat x$는 A행렬 각각의 column과 수직 관계에 있기 때문에 내적 값이 0이어야 한다.
$$
a_1^T(b-A\widehat{x})=0, a_2^T(b-A\widehat{x})=0\\\\
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



## 3. $A^TA$는  언제 역행렬이 존재하는가? 

- $A$의 column이 independent할 때 역행렬이 존재한다. (full rank)
- 왜? $A$ 와 $A^TA$의 null space가 같아서

(증명)

1. $N(A)\ni x \Rightarrow N(A^TA)\ni x$

$$Ax=0 \rightarrow  A^TAx=0$$

2. $N(A^TA)\ni x \Rightarrow N(A)\ni x $

$$A^TAx=0 \rightarrow  x^TA^TAx=0 = (Ax)^T(Ax)=0 \rightarrow Ax=0$$

$A$ 가 full rank인 경우에 A의 null space가 trivial하다. 위에서 $A$ 와 $A^TA$의 null space가 같다는 것을 증명했기 때문에  $A^TA$의 null space또한 trivial할 것이다. 이 말을 정리하면 $A$가 가역행렬인 경우에 $A^TA$도 가역행렬이며,  Least square problem은 $\hat{x}=(A^TA)^{-1}Ab$ 로 unique solution을 가질 것이다. 