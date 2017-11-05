# Liner Regression
## 简述
在统计学中，线性回归(Linear Regression)是利用称为线性回归方程的最小平方函数对一个或多个自变量和因变量之间关系进行建模的一种回归分析。这种函数是一个或多个称为回归系数的模型参数的线性组合。
机器学习中，我们通过线性回归得到的模型对其它的输入值预测出相对应的输出值。上述的模型我们称它为 假设，函数表达为 $h(x):\mathcal{X}\mapsto\mathcal{Y}$

## 基本过程
现在，我们有 $m$ 个样本 $\pmb{x}$,对于每一个样本 $\pmb{x}^{(i)}$ ，一共有 $n$ 个特征值，为 $\pmb{x}^{(i)}_{j}$ 。有 $n$ 个系数（模型参数）$\theta_0\quad\theta_1\dots\theta_n$。
则我们的预测函数为:
$$
h_{\theta}(x) = \theta_0 +\theta_1 x_1+\theta_2 x_2 +\dots+\theta_n x_n
$$

为了简单化函数的表示，我们规定 $x_0 = 1$，因此函数中的的参数 $\theta_0$ 就是参数 $\mathcal{bais}$ 。

上面提到的$\mathcal{bais}$，它的的作用会在以后的笔记中详细解释。
在计算机中，为了简便计算，我们将数据向量化（vectorize）：
$$
h(\pmb{x}) = \sum^{n}_{i=0} \theta_i x_i = \pmb{\theta}^{\mathit{T}} \pmb{x},\quad \pmb{\theta} \in \mathbb{R}^{n+1}, \pmb{x} \in \mathbb{R}^{n+1}
$$
在周志华的《机器学习》一书中，则将$h_{\theta}(\pmb{x})$，表示为：
$$
f(\pmb{x})=\pmb{w}^T\pmb{x}+b,\quad \pmb{w}\in \mathbb{R}^n,\pmb{x}\in \mathbb{R}^n,
$$
其实两个函数是一样的，因为有$\theta_0 = b$。但是在计算机中，第一种向量表示比较方便储存数据，我的笔记也就这么记了。

为了使估计值$\hat{y}$接近于$y$，我们定义**cost function** :
$$
\mathit{J}(\theta) = \frac{1}{2}\sum ^{m}_{i = 1}(h_{\theta}(x^{(i)})-y^{(i)})^2
$$
有了cost function，我们的目标就是：
$$
\DeclareMathOperator*{\argmin}{arg\,min} 
\pmb{\theta} = \mathop{\argmin}_{}{\mathit{J}_{\theta}(x)}
$$
## 求解 $\theta$ 的方法
> 基于均方误差最小化来进行模型求解的方法称为“最小二乘法”。——周志华《机器学习》

上述的 cost function 就是基于了均方误差。最小二乘法就是试图找到一条“线”，使所有样本到直线的欧式距离之和最小。

求解 $\pmb{\theta}$ 的方法有两个，一个是梯度下降（gradient descent），一个是通过求导得出极值。

### 梯度下降（gradient descent） 

首先，梯度下降是通过不断更新 $\theta$ 的值，而找到最优的情况。对于一个凸函数来说，是让当前的 $\theta $ 往函数下降最快的方向进行移动，以此更新 $\theta $ ，而更新的量，就是函数的导数了。所以，对于某一特征系数有更新公式：
$$
\theta_j := \theta_j-\alpha \cdot\frac{\partial}{\partial\theta_j}\mathit{J}(\theta)
$$
我们从一个样本来看梯度下降，首先对 $\mathit{J}(\theta)$ 求 $\theta$ 的偏导：
$$
\begin{equation}
\begin{aligned}
\frac{\partial}{\partial\theta_j}\mathit{J}(\theta) &= \frac{\partial}{\partial\theta_j}\frac{1}{2}(h_{\theta}(x)-y)^2\\
&=2\cdot\frac{1}{2}(h_{\theta}(x)-y)\cdot \frac{\partial}{\partial\theta_j}(h_{\theta}(x)-y)\\
&=(h_{\theta}(x)-y)\cdot \frac{\partial}{\partial\theta_j}(\sum^n_{i=0}\theta_ix_i-y)\\
&=(h_{\theta}(x)-y)x_j
\end{aligned} 
\end{equation}
$$
于是，更新公式可写为：
$$
\begin{aligned}
\theta_j &:= \theta_j-\alpha \cdot(h_{\theta}(x)-y)x_j\\
&:=\theta_j +\alpha \cdot(y-h_{\theta}(x))x_j
\end{aligned}
$$
对于一个训练集，我们就得到：
$$
\begin{eqnarray*}
&&\large{重复直到收敛}\{\\
&& \quad \quad \theta_j := \theta_j +\alpha \cdot \sum^{m}_{i=1}(y^{(i)}-h_{\theta}(x^{(i)}))x_j\\
 &&\}
\end{eqnarray*}
$$

### 求导

>   The "Normal Equation" is a method of finding the optimum theta without iteration.

首先，我们定义一下 $X$，$X$ 是一个 $m \times n$ 的矩阵，事实上，考虑截距（$\theta_0$），矩阵应该为  $m \times (n+1)$ ，

我们有：
$$
X = \left[
 \begin{matrix}
-(x^{(1)})^T-\\
-(x^{(2)})^T- \\
\vdots\\
-(x^{(m)})^T- 
  \end{matrix}
  \right]
$$
然后有 $\pmb{y}$  ：
$$
\pmb{y} = \left[
 \begin{matrix}
y^{(1)}\\
y^{(2)}\\
\vdots\\
y^{(m})
  \end{matrix}
  \right],\pmb{y} \in \mathbb{R}^m
$$
于是有：
$$
J(\pmb{\theta}) = \frac{1}{2}(X\pmb{\theta}-\pmb{y})^T(X\pmb{\theta}-\pmb{y})
$$
因此，对 $\mathit{J}(\theta)$ 求 $\theta$ 的偏导：
$$
\begin{aligned}
\nabla_{\theta} J(\pmb{\theta}) &= \nabla_{\theta} \frac{1}{2}(X\pmb{\theta}-\pmb{y})^T(X\pmb{\theta}-\pmb{y})\\
& = \frac{1}{2}\nabla_{\theta} (\pmb{\theta}^TX^TX\pmb{\theta}-\pmb{\theta}^TX^T\pmb{y}-\pmb{y}^TX\pmb{\theta}+\pmb{y}^T\pmb{y})\\
& =\frac{1}{2}\nabla_{\theta} tr (\pmb{\theta}^TX^TX\pmb{\theta}-\pmb{\theta}^TX^T\pmb{y}-\pmb{y}^TX\pmb{\theta}+\pmb{y}^T\pmb{y})\\
& =  \frac{1}{2}\nabla_{\theta}(tr \pmb{\theta}^TX^TX\pmb{\theta} - 2 tr \pmb{y}^TX\pmb{\theta})\\
&=\frac{1}{2} (X^TX\pmb{\theta}+X^TX\pmb{\theta}-2X^T\pmb{y})\\
&=X^TX\pmb{\theta}-X^T\pmb{y}
\end{aligned}
$$
当对 $\theta$ 的偏导为 $0$ 时，可得极值，得：
$$
X^TX\pmb{\theta} = X^T\pmb{y}
$$
得到最优 $\theta$ 解 ：
$$
\pmb{\theta} = (X^TX)^{-1}X^T\pmb{y}
$$

## 概率解释（Probabilistic Interpretation）

### 简单说明

在对数据进行概率假设的基础上，最小二乘回归得到的 $\theta$ 和最大似然法估计的 $\theta$ 是一致的。所以这是一系列的假设，其前提是认为最小二乘回归（least-squares regression）能够被判定为一种非常自然的方法，这种方法正好就进行了最大似然估计（maximum likelihood estimation）。

### 证明

首先假设目标变量与输入变量存在以下等量关系：
$$
\pmb{y}^{(i)}  = \pmb{\theta}^T\pmb{x}+ \pmb{\varepsilon}^{(i)}
$$
上式的 $\pmb{\varepsilon}^{(i)}$ 是误差项，用于存放由于建模所忽略的变量导致的效果 (比如可能某些特征对于房价的影响很明显，但我们做回归的时候忽略掉了)或者随机的噪音信息（random noise）。进一步假设 $\pmb{\varepsilon}^{(i)}$ 是独立同分布的 (IID ，independently and identically distributed) ，服从高斯分布（Gaussian distribution），其平均值为 $0$，方差（variance）为 $\sigma^2$。这样就可以把这个假设写成 $\pmb{\varepsilon}^{(i)} \sim \mathcal{N}(0,\sigma^2)$。然后 $\pmb{\varepsilon}^{(i)}$ 的密度函数就是：
$$
p(  \pmb{\varepsilon}^{(i)} ) = \frac{1}{\sqrt{2\pi}\sigma}\mathrm{exp}(-\frac{(\varepsilon^{(i)})^2}{2\sigma^2})
$$
这意味着存在下面的等量关系：
$$
p(\pmb{y}^{(i)}|\pmb{x}^{(i)};\pmb{\theta }) = \frac{1}{\sqrt{2\pi}\sigma}\mathrm{exp}(-\frac{(y^{(i)}-\pmb{\theta}^T\pmb{x}^{(i)})^2}{2\sigma^2})
$$
上式为，在 $\theta$ 取某个固定值的情况下，这个等式表达为，在 $\pmb{x}^{(i)}$ 的情况下发生 $\pmb{y}^{(i)}$ 的概率， 通常可以看做是一个 $\pmb{y}^{(i)}$ 的函数。当我们要把它当做 $\theta$ 的函数的时候，就称它为似然函数（likelihood function）在整个数据集下有：
$$
L(\pmb{\theta}) = L(\pmb{\theta};X,\pmb{y}) = p(\pmb{y}|X;\pmb{\theta })
$$
结合之前对 $ \pmb{\varepsilon}^{(i)}$ 的独立性假设（这里对 $\pmb{y}^{(i)}$ 以及给定的 $\pmb{x}^{(i)}$ 也都做同样假设），就可以把上面这个等式改写成下面的形式： 
$$
\begin{align}
L(\pmb{\theta}) &= \prod^m_{i=1}(\pmb{y}^{(i)}|\pmb{x}^{(i)};\pmb{\theta }) \\
&= \prod^m_{i=1} \frac{1}{\sqrt{2\pi}\sigma}\mathrm{exp}(-\frac{(y^{(i)}-\pmb{\theta}^T\pmb{x}^{(i)})^2}{2\sigma^2})
\end{align}
$$
最大似然法（maximum likelihood）告诉我们要选择能让数据的似然函数尽可能大的 $\theta$ 。也就是说，找到  $\theta$ 能够让函数 $L(\pmb{\theta})$ 取到最大值。

为了找到 $L(\pmb{\theta})$ 的最大值，我们不能直接使用 $L(\pmb{\theta})$ ，而要使用严格递增的 $L(\pmb{\theta})$ 的函数求最大值。使用对数函数来找对数函数 $L(\pmb{\theta})$ 的最大值是一种方法，而且求导来说就简单了一些：
$$
\begin{align}
v& = \log{\mit{L}(\pmb{\theta})}\\
&=\log{ \prod^m_{i=1} \frac{1}{\sqrt{2\pi}\sigma}\mathrm{exp}(-\frac{(y^{(i)}-\pmb{\theta}^T\pmb{x}^{(i)})^2}{2\sigma^2})}\\
&=\sum^m_{i=1}\log{ \prod^m_{i=1} \frac{1}{\sqrt{2\pi}\sigma}\mathrm{exp}(-\frac{(y^{(i)}-\pmb{\theta}^T\pmb{x}^{(i)})^2}{2\sigma^2})}\\
&=m\log{ \frac{1}{\sqrt{2\pi}\sigma} -\frac{1}{\sigma^2} \cdot \sum^m_{i=1}(y^{(i)}-\pmb{\theta}^T\pmb{x}^{(i)})^2}
\end{align}
$$
对于上式 ，由于$m\log{\frac{1}{\sqrt{2\pi}\sigma}}$ ，$\frac{1}{\sigma^2}$ 值不变，那么 $\scr{L}(\pmb{\theta})$ 取最大，即求下面的式子最小：
$$
\frac{1}{2}\sum^m_{i=1}(y^{(i)}-\pmb{\theta}^T\pmb{x}^{(i)})^2
$$
证毕。

## 多项式回归（Polynomial Regession）

多项式回归可以用来拟合二次、三次、高次模型，通过使用 $\pmb{x}^2,\sqrt{\pmb{x}}$ 等进行拟合。 

这样便将高阶方程模型转换成线性回归模型。这也算是 **特征缩放(Features Scaling)** 的一种。
