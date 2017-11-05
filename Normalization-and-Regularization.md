# Normalization and Regularization

## 简述

Normalization 就是把数据进行前处理，从而使数值都落入到统一的数值范围，从而在建模过程中，各个特征量没差别对待。normalization 一般是把数据限定在需要的范围，比如一般都是 $\{0,1\}$ ，从而消除了数据量纲对建模的影响。并且对基于 gradient descent 的算法友好，能加快训练速度，促进算法的收敛。

注：Standardization 是 Normalization 的一种特殊情况，它对数据进行正态化处理，使数据的平均值为1，方差为0。

Regularization 是在 cost function 里面加惩罚项，增加建模的模糊性，从而把捕捉到的趋势从局部细微趋势，调整到整体大概趋势。虽然一定程度上的放宽了建模要求，但是能有效防止过拟合（over-fitting）的问题。  

## Normalization 

Normalization 的手段很多，主要有：

1.  min-max normalization:  $x'=\frac{x-min}{max-min}$ .
2.  logarithmic transformations：$x' = \frac{\log{(x)}} {\log{(x_{max})}}$ .
3.  arctan function：$x'= arctan(x)$ .
4.  zero mean normalization：$x'=\frac{x-\mu}{\sigma}$

## Regularization

Regularization 主要是处理过拟合的情况，它对某些特征值进行处罚，简单来说就是降低重要性。

以 Liner Regression 为例，在算法中，我们的步骤为：

cost function：
$$
J(\theta)=\frac{1}{2m}\left[ \sum^m_{i=1}(h_\theta(x^{(i)})-y^{(i)} )+\lambda\sum^n_{j=1}\theta^2_j \right]
$$
求导完后有梯度下降迭代式，其中 $x_0$ 始终为 1 ，不参与迭代：
$$
\begin{eqnarray*}
&& \large{重复直到收敛}\{\\
&& \quad \quad  \theta_0:= \theta_0-\alpha \cdot \frac{1}{m}\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-(y^{(i)})x^{(i)}_0 \\
&& \quad \quad \theta_j := \theta_j -\alpha \cdot \left[ \left( \frac{1}{m}\sum^{m}_{i=1}(h_{\theta}(x^{(i)})-(y^{(i)})x^{(i)}_j  \right) +\frac{\lambda}{m}\theta_j\right]\\
 &&\}
\end{eqnarray*}
$$
