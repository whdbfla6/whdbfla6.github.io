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



## 2. Gram-schmidt 

그람슈미트는 부분공간을 span하는 선형독립 벡터들을 orthonormal vector로 변환하는 과정으로, [projection]()

- projection을 이용해서 구할 수 있다
- $R^n$ 의 subspace $w$ basis가 {$a_1, \cdot , a_n$}라 하자

$$
q_1 = a_1/ \parallel a_1 \parallel\\ 
A_2 = a_2 -(a_2^Tq_1)q_1,\ q_2 = A_2/ \parallel A_2 \parallel\\ 
A_3 = a_3 -(a_3^Tq_1)q_1 -(a_3^Tq_2)q_2,\ q_3 = A_3/ \parallel A_3 \parallel\\
$$

- 각 q 벡터의 크기는 1이고, 서로 다른 q벡터의 내적은 0이다
- 그람슈미트를 통해 구한 {$q_1, \cdot , q_n$}를 orthonormal basis라 한다

## 3. QR 분해

![](http://whdbfla6.github.io/assets/linear-algebra/img9.PNG)

- Q는 orthonormal basis로 구성된 행렬
- $R = Q^TA$ $r_ij = q_i^Ta_j$로 상삼각행렬이다