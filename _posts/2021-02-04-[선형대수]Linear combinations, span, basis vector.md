---
title: '[선형대수]Linear combinations, span, basis vector'
use_math: true
comments: true
layout: single
classes: wide
categories:
  - 선형대수
tags:
  - 선형대수
---

## 1. 벡터에 대한 이해

벡터는 크게 물리학적인 관점, 컴퓨터과학 관점, 수학자들의 관점으로 나눠서 생각해 볼 수 있다. 물리학 관점에서 벡터는 길이와 방향을 가지며, 길이와 방향이 같다면 동일한 벡터다. 

컴퓨터 과학 관점에서는 순서가 있는 숫자 리스트다. 다음과 같이 $\begin{bmatrix}
2,600ft \\ 300,000
\end{bmatrix}$ 주택 가격을 분석하기 위해 면적 가격 두가지 요소를 고려하고 있다면 그 각각의 값을 리스트에 넣어 벡터로 표현할 수 있을 것이다. 

수학자들의 관점에서는 덧셈과 상수 곱이 보존되는 경우에 벡터라 한다. $V$가 nonempty set이고 $R$을 실수 공간인 경우 다음 두가지 조건을 만족하는 경우에 V를 **vector space**라 한다

- $u,v\in V \Rightarrow u+v\in V$
- $a\in R, v\in V \Rightarrow av\in V$

vector space에는 다양한 형태가 존재하지만 앞으로는 vector space의 대표적인 예시인 $R^n$공간만 고려할 것이다. Real Euclidean n-space $R^n$이 vector space인 이유는 다음과 같다

$$ R^n = \begin{bmatrix}
x_1 \\ x_2 \\ \cdots \\x_n
\end{bmatrix}: x_1,x_2,\cdots,x_n \in R $$

1. $$\alpha \in R, x\in R^n \Rightarrow \alpha x=\alpha \begin{bmatrix}
   x_1 \\ x_2 \\ \cdots \\x_n
   \end{bmatrix}  = \begin{bmatrix}
   \alpha x_1 \\ \alpha x_2 \\ \cdots \\ \alpha x_n
   \end{bmatrix} \in R^n$$
2. $$ x,y\in R^n \Rightarrow x+y =\begin{bmatrix}
   x_1 \\ x_2 \\ \cdots \\x_n
   \end{bmatrix} + \begin{bmatrix}
   y_1 \\ y_2 \\ \cdots \\y_n
   \end{bmatrix} = \begin{bmatrix}
   x_1+y_1 \\ x_2+y_2 \\ \cdots \\x_n
   +y_n\end{bmatrix} \in R^n $$

## 2.  Linear combinations, span, basis vector

>  Linear combination(선형결합) , span

벡터 $v,w$의 **span**은 두 벡터의 linear combination인 $av+bw$가 만들어내는 모든 집합을 의미한다.  예를들어 $\begin{bmatrix}
0 \\ 3
\end{bmatrix}, \begin{bmatrix}
2 \\ -1
\end{bmatrix}$ 두개의 벡터가 있다고 하자. 두 벡터의 linear combination이 만들어내는 공간은 무엇일까? 두 벡터의 선형결합은 $R^2$공간 전체를 채울 수 있다. 그렇다면 $\begin{bmatrix}
2 \\ 4
\end{bmatrix}, \begin{bmatrix}
1 \\ 2
\end{bmatrix}$ 의 경우도 $R^2$ 공간 전체를 채울 수 있을까? 이 경우에는 각 벡터에 스칼라곱을 하고 더하더라도 하나의 직선을 생성하는 것에 그치기 때문에, $R^2$를 span하지 않는다.



> Linearly independent

위에서 살펴본  $\begin{bmatrix}
2 \\ 4
\end{bmatrix}, \begin{bmatrix}
1 \\ 2
\end{bmatrix}$  의 경우 한 벡터가 나머지 벡터의 스칼라 곱 형태다. 이처럼 하나의 벡터가 나머지 벡터의 선형결합으로 표현할 수 있는 경우에 **선형 종속(linearly dependent)** 이라고 한다. 선형독립은 그 반대의 경우인데 정의는 다음과 같다.

벡터 $v_1,v_1,\cdots v_n$ 에 대해 linear combination $a_1v_1 + a_2v_2 +\cdots +a_nv_n=0$을 만족하는 경우가  $a_1=a_2=\cdots=a_n=0$인 경우만 존재할 때 **linearly independent**라 한다. 

$$a_1 \begin{bmatrix}
0 \\ 3
\end{bmatrix} + a_2 \begin{bmatrix}
2 \\ -1
\end{bmatrix} =0 \Rightarrow a_1=a_2=0 \cdots (a)$$

$$a_1 \begin{bmatrix}
2 \\ 4
\end{bmatrix} + a_2 \begin{bmatrix}
1 \\ 2
\end{bmatrix} =0 \Rightarrow a_1=-1, a_2=2 \cdots (b)$$

(b)의 경우에 0이 아닌 해가 존재하기 때문에 linearly dependent 하다.



> basis

Vector space의 **basis**는 그 공간을 span하는 선형독립인 벡터들의 집합이다.  $\begin{bmatrix}
0 \\ 3
\end{bmatrix}, \begin{bmatrix}
2 \\ -1
\end{bmatrix}$ 두개의 벡터는 선형 독립관계에 있으면서 $R^2$ 공간을 span하기 때문에 $R^2$의 기저로 볼 수 있다. 그렇다면 $\begin{bmatrix}
0 \\ 1 \\ 2
\end{bmatrix}, \begin{bmatrix}
1 \\ 0 \\ 0 
\end{bmatrix}, \begin{bmatrix}
0 \\ 2 \\ 4 
\end{bmatrix}$ 3개의 벡터는 $R^3$의 기저가 될 수 있을까? 이 경우에 하나의 벡터가 나머지 벡터의 상수배이기 때문에 3개의 벡터는 $R^3$ 공간에서 평면을 만드는데 그친다. $\begin{bmatrix}
1 \\ 0 \\ 0
\end{bmatrix}, \begin{bmatrix}
0 \\ 1 \\ 0 
\end{bmatrix}, \begin{bmatrix}
0 \\ 0 \\ 1 
\end{bmatrix}$는 선형독립이면서 $R^3$공간 전체를 채울 수 있기 때문에 basis이며, 이러한 형태는 basis 중에 x축 y축 z축에 해당하는 기본적인 형태이기 때문에 **standard basis**라 부른다.

- basis는 왜 중요한 것일까?

벡터공간에 있는 모든 벡터들은 basis의 선형결합으로 표현할 수 있기 때문에, **좌표만 알더라도 벡터를 표현**할 수 있다. 예를들어 $R^2$ 공간의 모든 벡터들은 standard basis로 표현이 가능하다. 
$$
\begin{bmatrix} x_1 \\ x_2\end{bmatrix} = x_1 \begin{bmatrix}1 \\ 0 \end{bmatrix}+ x_2 \begin{bmatrix}0 \\ 1\end{bmatrix}
$$
우리가 보편적으로 $(1,3)$이라 할 때 각 1과 3은 standard basis로 표현한 좌표를 의미한 것이었다. 여기서 **좌표**를 수식 기호로 표현하면 아래와 같다. 

$\mathcal{X}= \begin{Bmatrix} x_1, & x_2 \end{Bmatrix}$ 가 $R^2$공간의 basis라고 하자  $\begin{bmatrix} x\end{bmatrix}_{\mathcal{X}}$ 는 **basis $x_1,x_2$ 로 나타낸 벡터 $x$ 의 좌표**를 의미한다. 

예를들어 $\begin{bmatrix} 1 \\ 3\end{bmatrix}$ 벡터를 standard basis $\begin{bmatrix}1 \\ 0 \end{bmatrix}, \begin{bmatrix}0 \\ 1\end{bmatrix}$ 로 나타내보자. 

$$x =\begin{bmatrix} 1 \\ 3\end{bmatrix} = 1 \begin{bmatrix}1 \\ 0 \end{bmatrix}+ 3 \begin{bmatrix}0 \\ 1\end{bmatrix}$$

$$\begin{bmatrix} x\end{bmatrix}_{\mathcal{X}} = \begin{bmatrix}1 \\ 3 \end{bmatrix}$$

이번에는 standard basis가 아닌   $\begin{bmatrix}1 \\ -1 \end{bmatrix}, \begin{bmatrix}0 \\ 4\end{bmatrix}$  basis를 이용해 $\begin{bmatrix} 1 \\ 3\end{bmatrix}$을 표현해 보자

$$\begin{bmatrix} 1 \\ 3\end{bmatrix} = 1 \begin{bmatrix}1 \\ -1 \end{bmatrix}+ 1 \begin{bmatrix}0 \\ 3\end{bmatrix}$$

$$\begin{bmatrix} x\end{bmatrix}_{\mathcal{X}} = \begin{bmatrix}1 \\ 1 \end{bmatrix}$$

모든 벡터는 벡터공간의 basis로 표현할 수 있기 때문에 basis를 안다는 것은 벡터 공간 전체를 이해한 것과 같다고 볼 수 있다. 또한 동일한 벡터를 어떤 basis를 기준으로 잡는지에 따라 다른 좌표로 표현할 수 있다. 기저의 변환에 대한 이해는 향후 행렬의 대각화나 SVD 분해를 이해하는데 밑거름이 될 것이다.

