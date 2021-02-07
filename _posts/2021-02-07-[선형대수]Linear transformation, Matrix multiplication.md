---
title: '[선형대수]Linear transformation, Matrix, determinant'
use_math: true
comments: true
layout: single
classes: wide
categories:

  - 선형대수

tags:

  - 선형대수
---

## 1. Linear transformation(선형변환)

**Linear transformation**은 일종의 함수로 행렬에서의 선형변환은 벡터에 행렬을 곱해 또 다른 벡터로 변환하는 과정을 의미한다. 

$$
\begin{bmatrix}1 & -3  \\ 2 & 4  \end{bmatrix} \begin{bmatrix}5\\ 7  \end{bmatrix} = \begin{bmatrix}-16\\ 38  \end{bmatrix} \\
\begin{bmatrix}5\\ 7  \end{bmatrix} \Rightarrow  \begin{bmatrix}-16\\ 38  \end{bmatrix} \\
$$

> 선형변환의 정의

두 벡터공간 $V,W$에 대해 $T : V\rightarrow W$함수가 있다고 하자. 다음 조건을 만족할 때 $T$를 선형 변환이라고 한다

- $u,v\in V \Rightarrow T(u+v) = T(u)+T(v)$
- $a\in R, v\in V \Rightarrow T(av)=aT(v)$



## 2. Matrix: 행렬에 대한 이해

$R^2$공간에서 표준기저벡터 $$i = \begin{bmatrix}1 \\ 0 \end{bmatrix} j=\begin{bmatrix}0 \\ 1 \end{bmatrix}$$ 가 각각 $$\begin{bmatrix}3 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 2 \end{bmatrix}$$​로 이동할 때 임의의 벡터 $$\begin{bmatrix}x \\ y \end{bmatrix}$$는 어떤 벡터로 이동할 것인가? 이에 대한 해답이 행렬이라고 볼 수 있다. 


$$
A = \begin{bmatrix}3 & 1  \\ 0 & 2  \end{bmatrix}\\ Ax = \begin{bmatrix}3 & 1  \\ 0 & 2  \end{bmatrix} \begin{bmatrix}x \\ y  \end{bmatrix} \rightarrow 벡터 \begin{bmatrix}x \\ y  \end{bmatrix}는\ 어디로\ 갈까?
$$


조금 더 자세히 살펴보자.  행렬 $A$에 $$\begin{bmatrix}1 \\ 1 \end{bmatrix}$$ 를 곱하면 $$\begin{bmatrix}4\\ 2\end{bmatrix}$$​가 된다. $$\begin{bmatrix}1 \\ 1 \end{bmatrix}$$ 이 아닌 임의의 벡터 $$\begin{bmatrix}x \\ y \end{bmatrix}$$ 에 $A$를 곱한다면 각 벡터에 상응하는 출력벡터를 얻을 수 있을 것이다. 즉 2X2 행렬의 곱은 $R^2$공간에 놓인 모든 벡터들을 다른 벡터로 변환하는 **공간의 이동**으로 이해할 수 있다. 

![img](http://whdbfla6.github.io/assets/linear-algebra/img19.png)

[그림출처](https://shad.io/MatVis/)

벡터 $$v=\begin{bmatrix}1 \\ 2 \end{bmatrix}$$는 기저를 $$\begin{bmatrix}1 \\ 0 \end{bmatrix},\begin{bmatrix}0 \\ 1 \end{bmatrix}$$ 로 바라볼 때 각각의 좌표를 의미하기 때문에 $$v=1i+2j$$로 나타낼 수 있다. 따라서 각 기저벡터가 다른 벡터로 이동한다면 벡터 $v$는 $-1(transformed\ i)+2(transformed\ j)$로 이동할 것이다.  

- $i,j$가 각각 $$\begin{bmatrix}3 \\ 0 \end{bmatrix}, \begin{bmatrix}1 \\ 2 \end{bmatrix}$$로 이동했다고 가정해보자


$$
(transformed)v=-1(transformed\ i)+2(transformed\ j)\\
(transformed)v=-1\begin{bmatrix}3 \\ 0 \end{bmatrix}+2\begin{bmatrix}1 \\ 2 \end{bmatrix}
$$


$$-1\begin{bmatrix}3 \\ 0 \end{bmatrix}+2\begin{bmatrix}1 \\ 2 \end{bmatrix}$$는 $$\begin{bmatrix}3 & 1  \\ 0 & 2  \end{bmatrix} \begin{bmatrix}-1 \\ 2\end{bmatrix}$$와 같은 형태이며 벡터 $$\begin{bmatrix}-1 \\ 2\end{bmatrix}$$ 가 아닌 임의의 벡터 $$\begin{bmatrix}x \\ y\end{bmatrix}$$ 의 이동을 알고 싶다면 $$\begin{bmatrix}3 & 1  \\ 0 & 2  \end{bmatrix} \begin{bmatrix}x \\ y\end{bmatrix}$$​ 계산을 통해 쉽게 구할 수 있을 것이다. 



> 예시: 시계 반대 방향 90도 회전

$R^2$공간에 있는 기존 벡터들을 시계 반대 방향으로 90도 회전하고 싶다고 하자. 표준기저벡터는 각각 다음과 같이 이동할 것이다. 


$$
i = \begin{bmatrix}1 \\ 0 \end{bmatrix} \Rightarrow \begin{bmatrix}0 \\ 1 \end{bmatrix}\\ j = \begin{bmatrix}0 \\ 1 \end{bmatrix} \Rightarrow \begin{bmatrix}-1 \\ 0 \end{bmatrix}\\
$$
따라서 90도 회전(시계반대)에 대응하는 행렬은 다음과 같다



$$
A = \begin{bmatrix}0 & -1  \\ 1 & 0  \end{bmatrix}
$$


![img](http://whdbfla6.github.io/assets/linear-algebra/img20.png)

벡터 $$v=\begin{bmatrix}1 \\ 2\end{bmatrix}$$  을 90도 시계 반대 방향으로 회전한다고 하면 $$Av=\begin{bmatrix}0 & -1  \\ 1 & 0  \end{bmatrix} \begin{bmatrix}1 \\ 2\end{bmatrix} = \begin{bmatrix}-2 \\ 1\end{bmatrix}$$​ 로 변환될 것이다. 

## 3. determinant(행렬식)

앞서 2X2 행렬의 곱은 $R^2$공간에 놓인 모든 벡터들을 다른 벡터로 변환하는 **공간의 이동**이라고 했다. 이 때 **공간이 얼마나 확장했는지**를 나타낸 것이 **행렬식**이다

행렬 $$\begin{bmatrix}3 & 0  \\ 0 & 2  \end{bmatrix}$$에 대한 determinant는 공식에 따라 $ad-bc=6$임을 쉽게 구할 수 있다.  표준기저벡터 $$\begin{bmatrix}1 \\ 0 \end{bmatrix},\begin{bmatrix}0 \\ 1 \end{bmatrix}$$ 가 각각 $$\begin{bmatrix}3 \\ 0 \end{bmatrix},\begin{bmatrix}0 \\ 2 \end{bmatrix}$$ ​로 이동하면서 표준기저벡터가 이루던 단위공간 면적 1이 6으로 확장되었음을 확인할 수 있는데, 이 때 6이 Determinant 값이다.

![img](http://whdbfla6.github.io/assets/linear-algebra/img21.png)

> determinant < 0 

그렇다면 어떤 상황에서 행렬식이 음수값이 나올까? 표준기저벡터 $i$와 $j$의 순서가 바뀌어 공간 자체가 뒤집히는 경우에 행렬식은 음수가 된다. 

![img](http://whdbfla6.github.io/assets/linear-algebra/img22.png)

