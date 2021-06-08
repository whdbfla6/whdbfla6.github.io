---
title: '[선형대수]Orthogonal matrix, Gram-schmidt'
use_math: true
comments: true
layout: single
classes: wide
categories:
  - 선형대수
tags:
  - 선형대수
---

## 1. Orthogonal matrix: 직교행렬

> Orthonormal vector: 정규직교벡터

정규직교벡터는 두 벡터가 수직인 관계에 있으면서 크기가 1인 벡터다


$$
\boldsymbol{q}_{i}^{T} \boldsymbol{q}_{j}=\left\{\begin{array}{lll}
0 & \text { if } & i \neq j \\
1 & \text { if } & i=j
\end{array}\right.
$$

> orthogonal matrix: 직교행렬

행과 열이 orthonormal vector로 이루어진 행렬로, $Q^TQ$가 항등행렬이 된다는 성질이 있다


$$
Q=\left[\begin{array}{ccc}
\mid & & \mid \\
\boldsymbol{q}_{1} & \cdots & \boldsymbol{q}_{n} \\
\mid & & \mid
\end{array}\right]
$$


$$
Q^{T} Q=I \rightarrow\left[\begin{array}{ccc}- & \boldsymbol{q}_{1}^{T} & - \\& \vdots & \\ - & \boldsymbol{q}_{n}^{T} & -\end{array}\right]\left[\begin{array}{ll}\mid & & \mid \\ \boldsymbol{q}_{1} & \cdots & \boldsymbol{q}_{n} \\ \mid & & \mid \end{array}\right]=\left[\begin{array}{ccc} 1 & \cdots & 0 \\ \vdots & \ddots & \vdots \\ 0 & \cdots & 1 \end{array}\right]
$$


직교 행렬이 행과 열의 수가 같은 경우에 $Q^T=Q^{-1}$이 된다. 즉 행과 열의 위치를 바꿔서 쉽게 역행렬을 구할 수 있다

- 직교 행렬의 이점

[이전 포스트](https://whdbfla6.github.io/%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98/%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98-Linear-combinations,-span,-basis-vector/)에서 살펴보았듯이 직교행렬을 이용하면 투영행렬을 더 쉽게 구할 수 있다. 투영행렬 $P=A(A^TA)^{−1}A$ 에서 행렬 $A$가 직교행렬인 경우에 $Q(Q^TQ)^{−1}Q^T=QIQ^T=QQ^T$ 간단한 형태가 된다. 

## 2. Gram-schmidt 

그람슈미트는 부분공간을 span하는 <u>선형독립 벡터들을 orthonormal vector로 변환하는 과정</u>으로, [projection](https://whdbfla6.github.io/%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98/%EC%84%A0%ED%98%95%EB%8C%80%EC%88%98-Linear-combinations,-span,-basis-vector/) 과정을 이용해서 쉽게 구할 수 있다. 

<p align="center"><img src="http://whdbfla6.github.io/assets/linear-algebra/img18.png" style="zoom:80%;" /></p>



$R^n$ 의 subspace $w$ basis가 {$a_1, \cdots , a_n$}라 하자

1. 기존 $a_1$ 벡터를 $a_1$의 norm으로 나눠서 크기를 1로 바꿔준다.

2. $q_2$를 구하기 위해 $a_1$에 수직인 벡터인 $A_2$를 구한 후에 크기를 1로 만들어준다.

   $a_2$ 벡터를 $a_1$에 정사영한 후에 $a_2$에서 빼면 $a_1$에 수직인 벡터 $A_2$를 구할 수 있다. 그 후에 $A_2$ norm으로 나눠서 $q_2$를 구해준다. 
   
3. $q_3$벡터는 $a_3$벡터를 $a_1$과 $a_2$에 모두 수직인 벡터로 바꾼 후에 크기를 1로 만들어 준 것이다.

4. 이 과정을 반복하다보면 orthonormal basis {$q_1, \cdots , q_n$} 를 얻을 수 있다


$$
q_1 = a_1/ \parallel a_1 \parallel\\ 
A_2 = a_2 -(a_2^Tq_1)q_1,\ q_2 = A_2/ \parallel A_2 \parallel\\ 
A_3 = a_3 -(a_3^Tq_1)q_1 -(a_3^Tq_2)q_2,\ q_3 = A_3/ \parallel A_3 \parallel\\
$$



## 3. QR 분해

그람슈미트 과정을 통해 구한 $q$벡터를 이용해 다음과 같이 $QR$분해가 가능하다



$$
A = \left[\begin{array}{lll}
a_{1} & a_{2} & a_{3}
\end{array}\right]=\left[\begin{array}{lll}
q_{1} & q_{2} & q_{3}
\end{array}\right]\left[\begin{array}{ccc}
r_{11} & r_{12} & r_{13} \\
0 & r_{22} & r_{23} \\
0 & 0 & r_{33}
\end{array}\right] = QR
$$



$R = Q^TA$ 은 상삼각행렬로 대각성분 아래의 성분은 모두 0이다. $R$행렬을 구하는 과정은 다음과 같다. 


$$
\begin{bmatrix} q_1^T \\ q_2^T \\ q_3 ^T\end{bmatrix} \begin{bmatrix} a_1 & a_2 & a_3\end{bmatrix} = \left[\begin{array}{ccc}
q_1^Ta_1 & q_1^Ta_2 & q_1^Ta_3 \\
q_2^Ta_1 & q_2^Ta_2 & q_2^Ta_3 \\
q_3^Ta_1 & q_3^Ta_2 & q_3^Ta_3 
\end{array}\right] = \left[\begin{array}{ccc}
q_1^Ta_1 & q_1^Ta_2 & q_1^Ta_3 \\
0 & q_2^Ta_2 & q_2^Ta_3 \\
0 & 0 & q_3^Ta_3 
\end{array}\right]
$$


$$
\left[\begin{array}{ccc}
\mid & \mid & \mid & \mid \\
a_{1} & a_{2} & \cdots & a_{n} \\
\mid & \mid & \mid & \mid
\end{array}\right]=\left[\begin{array}{cccc}
\mid & \mid & \mid & \mid \\
\hat{q}_{1} & \hat{q}_{2} & \cdots & \hat{q}_{m} \\
\mid & \mid & \mid & \mid
\end{array}\right]\left[\begin{array}{cccc}
\boldsymbol{a}_{1}^{T} \boldsymbol{q}_{1} & \boldsymbol{a}_{2}^{T} \boldsymbol{q}_{1} & \cdots & \boldsymbol{a}_{n}^{T} \boldsymbol{q}_{1} \\
\boldsymbol{a}_{1}^{T} \boldsymbol{q}_{2} & \boldsymbol{a}_{2}^{T} \boldsymbol{q}_{2} & \cdots & \boldsymbol{a}_{n}^{T} \boldsymbol{q}_{2} \\
\vdots & \vdots & \ddots & \vdots \\
\boldsymbol{a}_{1}^{T} \boldsymbol{q}_{m} & \boldsymbol{a}_{2}^{T} \boldsymbol{q}_{m} & \cdots & \boldsymbol{a}_{n}^{T} \boldsymbol{q}_{m}
\end{array}\right]
$$



$q_2^Ta_1$를 살펴보면 $q_2$는 $a_1$에 직교하는 벡터이기 때문에 내적값이 0이 나온다. 비슷한 계산을 통해 대각선 아래 성분이 0이 되는 것을 확인할 수 있다.

