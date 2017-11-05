# Logistic Regression

## 简述
在 Liner Regression 中 $\pmb{y}$ 是一个连续值，那么当我们解决一个二分类问题时 $\pmb{y}$ 则是一个离散值，只有两个取值（ $0$ 或 $1$ ），这时可以通过广义线性模型来解决。通过广义线性模型，我们找到一个单调可微函数将分类任务的标记 $\pmb{y}$ 与线性模型的预测值联系起来。最后我们找到 Logistic Function 来作为线性模型。

## 广义线性模型（Generalized Linear Models ）

在笔记中，我先把广义线性模型梳理一遍，这有助于我更好得学习机器学习。

当然如果仅仅是为了学习 Logistic Function 那么可以先看看后面的内容，之后再看这部分内容。

在回归学习中，我们的函数都类似于$f(\pmb{y}|\pmb{x}); \pmb{\theta} \sim \mathcal{N}(\mu,\sigma^2)$ 或者在之后讲的二分类函数 $f(\pmb{y}|\pmb{x}); \pmb{\theta} \sim \rm{Bernoulli}(\phi)$ ，这里的 $\mu$ 和 $\phi$ 都分别是 $x$ 和 $\theta$ 的某种函数（ $\sigma$ 与分布无关 ）。其实，有一种更广泛的模型，这两种模型都是它的特例，这种更广泛的模型叫做广义线性模型。

>   在广义线性模型中(GLM), 对于每个独立参数的 $\pmb{y}$ ，假设通过一个指数族产生。这就是说，对于均值 $\mu$ 
>
>   , 和独立变量 $\pmb{x}$，有：
>   $$
>   E(\pmb{y})=\pmb{\mu}=g^{-1}(\pmb{\theta}^{\mathit{T}} \pmb{x})
>   $$
>   $E(\pmb{y})$ 是 $\pmb{y}$ 的期望；$\pmb{\theta}^{\mathit{T}} \pmb{x}$  是一个线性估计; $g$ 是链接函数。

关于广义线性模型更多的知识请前往 [Wikipedia](https://en.wikipedia.org/wiki/Generalized_linear_model#Model_components) 。

### 指数族（The Exponential Family ）

在学习广义线性模型之前，我们要先定义一下指数族分布（exponential family distributions）。如果一个分布能用下面的方式来写出来，我们就说这类分布属于指数族： 
$$
p(y;\eta)=b(y)\rm{exp}(\eta^TT(y)-a(\eta))
$$
上面的式子中，

$\eta$ ：该分布的自然参数（natural parameter，也叫典范参数 canonical parameter）；

$T(y)​$ ：充分统计量（sufficient statistic），我们目前用的这些分布中通常 $T(y) = y​$ ；

 $a(\eta)$ ：一个**对数分割函数（log partition function）**；

$e^{−a(\eta)} $ ：这个量本质上扮演了归一化常数（normalization constant）的角色，也就是确保分布的 $p(y;\eta)$ 的总和等于1。

对 给定的 $T$ , $a$ 和 $b$ 就定义了一个以 $\eta$ 为参数的分布族（family，或者叫集 set）；通过改变 $\eta$ ，我们就能得到这个分布族中的不同分布。 

现在咱们看到的伯努利（Bernoulli）分布和高斯（Gaussian）分布就都属于指数分布族。伯努利分布的均值是 $\phi$ ，也写作 $\rm{Bernoulli}(\phi)$ ，确定的分布是 $y \in\{0,1\}$，因此有 $p(y=1;\phi)=\phi;p(y=0;\phi)=1-\phi$。这时候只要修改 $\phi$ ，就能得到一系列不同均值的伯努利分布了。现在我们展示的通过修改 $\phi$ ,而得到的这种伯努利分布，就属于指数分布族；也就是说，只要给定一组 $T$ , $a$ 和 $b$ ，就可以用上面的等式来确定一组特定的伯努利分布了。

伯努利分布通过广义线性模型可以这样写：
$$
\begin{align}
p(y;\phi) &= \phi^y(1-\phi)^{1-y}\\
&=\rm{exp}(\it{y}\rm{}\log{\phi}+(1-y)\log{(1-\phi)})\\
&=\rm{exp}\left(\left(\log{\left(\frac{\phi}{1-\phi}\right)}\right)\it{y}\rm{}+\log{(1-\phi)}\right)
\end{align}
$$

因此，给出了自然参数（natural parameter）$\eta=\log{\left(\frac{\phi}{1-\phi}\right)}$ 。 有趣的是，如果我们翻转这个定义，通过 $\phi$ 表示 $\eta$ 就会得到 $\phi=\frac{1}{1+e^{-\eta}}$ 。这正好就是我们之后在 Logistic Function 中会见到的 S 型函数（sigmoid function）！ 在我们把逻辑回归作为一种广义线性模型（GLM）的时候还会遇到伯努利分布以如下情况表示。
$$
\begin{align}
T(y) &= y\\
a(\eta)&=-\log{(1-\phi)}\\
&=log(1+e^\eta)\\
b(y)&=1
\end{align}
$$
接下来就看看高斯分布。在推导线性回归的时候， $\sigma^2$ 的值对我们最终选择的 $\theta $ 和 $h_\theta(x)$ 都没有影响。所以我们可以给 $\sigma^2$ 取一个任意值。为了简化推导过程，就令 $\sigma^2 = 1$ 。然后就有了下面的等式：
$$
\newcommand{\itm}[1]{\mathcal{#1}\rm}
\newcommand{\bgroup}[1]{\left({#1}\right)}
\begin{align}
p(y;\mu)&=\frac{1}{\sqrt{2\pi}}\rm{exp}\left(-\frac{1}{2}(\itm{y}\rm{}-\mu)\right)\\
&=\frac{1}{\sqrt{2\pi}}\rm{exp}\bgroup{-\frac{1}{2}\itm{y}^2}\cdot \rm{exp}\bgroup{\mu\itm{y}-\frac{1}{2}\mu^2}
\end{align}
$$

注：如果我们把 $\sigma^2$ 作为一个变量，高斯分布就也可以表达成指数分布的形式，其中 $\eta \in \mathbb{R}^2$ 就是一个同时依赖  $\mu$  和 $\sigma$ 的二维向量。然而，对于广义线性模型GLMs方面的用途，参数  $\sigma^2$ 就也可以看成是对指数分布族的更泛化的定义：$p(y;\eta,\tau)=b(a,\tau)\rm{exp}((\eta^T\it{T}(\itm{y})-\mathcal{a}(\eta))/\mathcal{c}(\tau))$ 。这里面的 $\tau$ 叫做分散度参数（dispersion parameter），对于高斯分布，$c(\tau)=\sigma^2$ ；不过上文中已经进行了简化，所以就不对各种需要考虑的情况进行更为泛化的定义了。

## Logistic Regression

### 基本过程

从最基本开始，我们不使用广义线性模型，对于二分类问题，输出标记 $\pmb{y} \in {0,1} $ ，于是我们使用线性回归最基本的模型 $z=  \pmb{\theta}^{\mathit{T}} \pmb{x}$ 来预测 $\pmb{y}$ ，即我们将 $z$ 转化为 $0/1$ 值。最理想的是 “单位越阶函数”（unit-step function）：
$$
\newcommand{\itm}[1]{\it{#1}\rm}
\newcommand{\bgroup}[1]{\left({#1}\right)}
y=\left\{
\begin{aligned}
0,& &z<0; \\
0.5,& &z=0; \\
1, & &z>0;
\end{aligned}
\right.
$$
但是，很明显，此函数不是很完美，于是，我们找到了一个“替代函数”来近似这个“单位跃阶函数”，并希望它单调可微，对数几率函数（Logistic Function）便满足这样一个条件：
$$
g(z)=\frac{1}{1+e^{-z}}
$$

![](https://ws3.sinaimg.cn/large/006tKfTcly1fl5zm1lftxj30jc0feq3p.jpg)

由图你能直观得看到，当 $z\to+\infty$ 的时候 $g(z)$ 趋向于 $1$ ，而当 $z\to-\infty$ 时 $g(z)$ 趋向于 $0$ 。

其实 $g(z)$ 也是 $h_\theta(x)$ ，且，像最开始一样，我们规定 $X_0 =1$ ，于是有：$\pmb{\theta}^Tx=\theta_0+\sum^{n}_{j=1}\theta_jx_j$ 

现在我们看看 $g'(z)$ 的特性：
$$
\begin{align}
g'(z) &= \frac{d}{dz}\frac{1}{1+e^{-z}}\\
&=\frac{1}{1+e^{-z}}(e^{-z})\\
&=\frac{1}{1+e^{-z}}\left(1-\frac{1}{1+e^{-z}}\right)\\
&=g(z)(1-g(z)).
\end{align}
$$
接着我们通过对 $h_\theta(x)$ 进行假设，得到：
$$
p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$
然后就能写出似然函数：
$$
L(\pmb{\theta}) = p(\pmb{y}|X;\pmb{\theta})
$$
又与之前一样写出 $L(\theta)$ 的对数函数 $\ell(\pmb{\theta})$ 以方便计算。

于是有 cost function：
$$
J(\theta) = -\frac{1}{m}\ell(\pmb{\theta})
$$
然后目标就是：
$$
\DeclareMathOperator*{\argmin}{arg\,min} 
\pmb{\theta} = \mathop{\argmin}_{}{J(\theta)}
$$

## $\pmb{\theta}$ 的求法

我们从最开始得到的假设函数讲起。如何得到它呢？

### 假设函数

我们首先假设：
$$
P(y=1|x;\theta)=h_\theta(x)\\
P(y=0|x;\theta)=1-h_\theta(x)
$$
于是，更简单的写法就是：
$$
p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}
$$

### 似然函数（likelihood function）

假设 $m$ 个训练样本都是各自独立的，那么就可以按如下的方式来写带参数的似然函数：
$$
\begin{align}
L(\pmb{\theta}) &= p(\pmb{y}|X;\pmb{\theta})\\
&=\prod^m_{i=1}p(y^{(i)}|x^{(i)};\pmb{\theta})  &将不同的样本的概率相乘\\  
&=\prod^m_{i=1}(h_\theta(x^{(i)}))^{y^{(i)}}(1-h_\theta(x^{(i)}))^{1-y^{(i)}}
\end{align}
$$
取对数更容易计算：
$$
\begin{align}
\ell(\pmb{\theta})&=\log{L(\theta)}\\
&=\sum^{m}_{i=1}y^{i}\log{h(x^{(i)})}+ (1-y^{i})\log{(1-h(x^{(i)}))}
\end{align}
$$

### Cost Function

极大似然函数中，为了求得最优的 $\pmb{\theta}$ ，就是让 $\ell(\pmb{\theta})$ 尽可能得大，于是在 cost function 中，我们加入系数 $-\frac{1}{m}$ ，于是得到了：
$$
J(\theta) = -\frac{1}{m}\sum^{m}_{i=1}y^{i}\log{h(x^{(i)})}+ (1-y^{i})\log{(1-h(x^{(i)}))}
$$


### 梯度下降法（gradient  descent）

其实，可以直接对 $\ell(\pmb{\theta})$ 做梯度上升求得 $\pmb{\theta}$ 。

按照向量的形式，我们对 $\theta$ 的更新可以写成：
$$
\pmb{\theta}:=\pmb{\theta}-\alpha \cdot \nabla_{\theta} J(\pmb{\theta})
$$
找到最优的第一步是对 $J(\pmb{\theta})$ 求导，我们从一个样本开始：
$$
\begin{align}
\frac{\partial}{\partial \theta_j}J(\theta)&=-\frac{1}{m}\ell(\theta)\\
&=-\frac{1}{m}\left( {y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)}}\right)\frac{\partial}{\partial\theta_j}g(\theta^Tx)\\
&=-\frac{1}{m}\left( {y\frac{1}{g(\theta^Tx)}-(1-y)\frac{1}{1-g(\theta^Tx)}}\right)g(\theta^Tx)(1-g(\theta^Tx))\\
&=-\frac{1}{m}\left(y(1-g(\theta^Tx))-(1-y)g(\theta^Tx)\right)x_j\\
&=\frac{1}{m}(h_\theta(x)-y)x_j
\end{align}
$$
其中，用到了上面提到的  $g'(z)$ 的特性，即 $\frac{\partial}{\partial\theta_j}g(\theta^Tx)=g(\theta^Tx)(1-g(\theta^Tx))$ ，然后梯度上升就简单写为：
$$
\pmb{\theta}:=\pmb{\theta}-\alpha \cdot \frac{1}{m} (y-h_\theta(x))x_j
$$
然后，再扩展为一个训练集：
$$
\begin{eqnarray*}
&&\large{重复直到收敛}\{\\
&& \quad \quad \theta_j := \theta_j +\alpha \cdot \frac{1}{m} \sum^{m}_{i=1}(y^{(i)}-h_{\theta}(x^{(i)}))x_j\\
 &&\}
\end{eqnarray*}
$$
有趣的是，这个式子正好与线性回归看上去一样，但是这实际上并不相同，原因是，我们对于  $h_\theta(x)$  的定义不同。但为什么相似呢？深层次的原因在于 **广义线性模型** 。

### $L(\theta)$ 最大的其它算法

下面这个方法更好，但是数学难度较高，其基本方法是“求方程零点的牛顿法”。

具体讲讲。假如我们有一个从实数到实数映射的函数 $\it{f}:\rm{R} \mapsto \rm{R}$，然后要找到一个 $\theta$ ，来满足 $\it{f}\,\rm{ }(\theta)=0$，其中$\theta \in R$是一个实数。牛顿法就是对 $\theta$ 进行如下的更新： 
$$
\theta:=\theta-\frac{f(\theta)}{f'(\theta)}
$$
对这个公式的简单解释是：通过一条逼近曲线的直线（切线）不断迭代来找到零点。

![](https://ws2.sinaimg.cn/large/006tKfTcly1fl66kyi55pj30co0b6q35.jpg)

对于上述的图，在 A 的切线方程为：$y_A = f(x_A)+f'(x_A)(x-x_A)$ 。

为了求下一个迭代点 B ，则有 $y_B = 0 = f(x_A)+f'(x_A)(x_B-x_A)$ ，

即：
$$
\begin{align}
&\Longrightarrow x_B=x_a-\frac{f(x_A)}{f'(x_A)}\\
&\Longrightarrow x_{n+1}=x_{n}-\frac{f(x_n)}{f'(x_n)}\\
&\Longrightarrow \theta:=\theta-\frac{f(\theta)}{f'(\theta)}
\end{align}
$$
用这个办法，我们求解 $\ell(\theta)$ 中 $\theta$ 的最优取值，即算 $\ell(\theta) = 0$ 的解，即可以通过以下迭代式求得：
$$
\theta:=\theta-\frac{\ell'(\theta)}{\ell''(\theta)}
$$
对于向量 $\pmb{\theta}$ 的求解，我们扩展牛顿法到多维的情况，叫做牛顿-拉普森法（Newton-Raphson method），如下：
$$
\pmb{\theta} := \pmb{\theta}-H^{-1}\nabla_{\theta} \ell(\pmb{\theta})
$$
其中 $\nabla_{\theta} \ell(\pmb{\theta})$ 是 $\ell(\pmb{\theta})$ 对 $\theta$ 的偏导。$H$ 是一个 $n\times n$ 矩阵（考虑 $\theta_0$ 的话是 $(n+1)\times (n+1)$  ），也可叫做 Hessian，具体定义为：
$$
H_{ij} = \frac{\partial^2\ell(\theta)}{\partial\theta_i\partial\theta_j}
$$
注意，当 $n$ 小的时候牛顿法的速度明显更快，但是当 $n$ 较大时，由于需要处理 Hession 矩阵，时间开销急剧增加。

## 构建广义线性模型

设想你要构建一个模型，来估计在给定的某个小时内来到你商店的顾客人数（或者是你的网站的页面访问次数），基于某些确定的特征 $X$ ，例如商店的促销、最近的广告、天气、今天周几啊等等。我们已经知道泊松分布（Poisson distribution）通常能适合用来对访客数目进行建模。知道了这个之后，怎么来建立一个模型来解决咱们这个具体问题呢？非常幸运的是，泊松分布是属于指数分布族的一个分部，所以我们可以使用一个广义线性模型（Generalized Linear Model，缩写为 ExpoFamilyGLM）。

对刚刚这类问题如何构建广义线性模型呢？

对于这类问题，我们希望通过一个 $X$ 的函数来预测 $\pmb{y}$ 的值。为了构建出模型，我们先给出3个假设：

1.  $y|x;\theta \sim \rm{ExponentialFamily(\eta)}$ 。即，给出 $x$ 和 $\eta$ ，则 $y$ 的分布遵循于指数分布。
2.  给出了 $x$ 我们的目标是预测 $T(y)$ 的期望值。大多数情况下 $T(y) = y$ ，也就是说，我们希望通过假设 $h$ 输出的 $h(x)$ 能满足 $h(x) = E[y|x]$ 。(统计学知识，有点难)  
3.  $\pmb{\eta}$ 和 $\pmb{x}$ 是线性相关的，

### 普通最小二乘（Ordinary Least Squares）

普通最小二乘其实是广义线性模型的一个特例，其中 $y$ 是连续的，通过 $x$ 给出的 $y$ 服从高斯分布 $\mathcal{N}(0,\sigma^2)$ ，经过上面的学习我们有：
$$
\begin{align}
h_{\theta}&=E[\pmb{y}|\pmb{x};\pmb{\theta}]\\
&=\pmb{\mu}\\
&=\pmb{\eta}\\
&=\pmb{\theta}^T\pmb{x}.
\end{align}
$$
第一行的等式是基于假设2；第二个等式是基于定理当 $y|x;\theta \sim \mathcal{N}(0,\sigma^2)$ ，则 y 的期望就是 μ；第三个等式是基于假设1，以及之前我们此前将高斯分布写成指数族分布的时候推导出来的性质 $\pmb{\mu}=\pmb{\eta}\\$ ；第四个等式就是基于假设3。

### Logistic Regression

二分类问题 $y\in \{0,1\}$ ，可以通过伯努利分布（Bernoulli distribution）来对给定 $x$ 的 $y$ 进行建模。

伯努利分布，有 $\phi=\frac{1}{1+e^{-\eta}}$ ，和在 $ y|x;\theta \sim \rm{Bernoulli}(\phi)$ 下有 $\quad E[y|x;\theta] = \phi$ 。

则有：
$$
\begin{align}
h_{\theta}&=E[\pmb{y}|\pmb{x};\pmb{\theta}]\\
&=\pmb{\phi}\\
&=\frac{1}{1+e^{-\pmb{\eta}}}\\
&=\frac{1}{1+e^{-\pmb{\theta}^T\pmb{x}}}.
\end{align}
$$
这就是为什么在 Logistic Function 中我们用 $\frac{1}{1+e^{-z}}$  做假设，即，一旦我们假设给定 $x$ 的 $y$ 的分布是伯努利分布，那么根据广义线性模型和指数分布族的定义，就会得出这个式子。

注：一个自然参数 $\eta$ 的函数 $g$ ，$g(\eta)=E[T(y)|\eta]$，这个函数叫做规范响应函数（canonical response function），它的反函数 $g^{-1}$ 叫做规范链接函数（canonical link function）。因此，对于高斯分布来说，它的规范响应函数正好就是识别函数（identify function）；而对于伯努利分布来说，它的规范响应函数则是 logistic function。

### Softmax Regression

对于多分类问题，有 $y\in \{1,2,\cdots,k\}$ ，通过多项式分布（multinomial distribution） 建模。

把多项式推出一个广义线性模型，首先把多项式分布用指数分布族进行描述。

我们给出  $k$ 个参数 $\phi_1,\cdots,\phi_k$ ，对应各自的输出值的概率，由于 $\sum^k_{i=1}\phi_i=1$ ，所以，有 $\phi_k =1- \sum^{k-1}_{i=1} \phi_i$ 。注意，$\phi_i$ 其实是 $p(u=i;\phi)$ 。现在该出 $T(y)$ ：
$$
T(1)=\left[\begin{matrix} 1\\0\\0 \\ \vdots\\0\end{matrix} \right],
T(2)=\left[\begin{matrix} 0\\1 \\0\\ \vdots\\0\end{matrix} \right],
T(3)=\left[\begin{matrix} 0\\0 \\ 1\\\vdots\\0\end{matrix} \right],\cdots,
T(k-1)=\left[\begin{matrix} 0\\0 \\0\\ \vdots\\1\end{matrix} \right],
T(k)=\left[\begin{matrix} 0\\0\\0 \\ \vdots\\0\end{matrix} \right]
$$
与之前不同，不再有$ T(y) = y$；然后，$T(y)$现在是一个 $k – 1$ 维的向量，而不是一个实数了。向量 $T(y)$ 中的第 i 个元素写成$(T(y))_i$ 。

给出一个记号：指示函数（indicator function），即 $1\{\cdot\}$ 。如果参数为真，则等于1；反之则等于0。

所以我们可以把 $T (y)$ 和 $y$ 的关系写成 $ (T(y))_i = 1\{y = i\}$。

现在把多项式写出指数分布族：
$$
\begin{align}
p(y;\theta) &= \phi^{1\{y=1\}}_{1} \phi^{1\{y=2\}}_{2} \cdots \phi^{1\{y=k\}}_{k}\\
&= \phi^{1\{y=1\}}_{1} \phi^{1\{y=2\}}_{2} \cdots \phi^{1-\sum^{k-1}_{i=1}1\{y=i\}}_{k}\\
&= \phi^{(T(y))_1}_{1} \phi^{(T(y))_2}_{2} \cdots  \phi^{1-\sum^{k-1}_{i=1}(T(y))_i}_{k}\\
&=\rm{exp}\left((T(y))_1\log{(\phi_1)}+(T(y))_2\log{(\phi_2)} + \cdots + \left(1-\sum^{k-1}_{i=1}(T(y))_i\right)\log{(\phi_k)}\right)\\
&=\rm{exp}\left((T(y))_1\log{(\phi_1/\phi_k)}+(T(y))_2\log{(\phi_2/\phi_k)} + \cdots + (T(y))_{k-1}\log{(\phi_{k-1}/\phi_k)} +\log{(\phi_k)}\right)\\
&=b(y)\rm{exp}(\eta^T\it{T}\rm{}\,(y)-a(\eta))
\end{align}
$$
其中:
$$
\begin{align}
\eta&= \left[\begin{matrix} \log{(\phi_1/\phi_k)}\\\log{(\phi_2/\phi_k)}\\ \vdots\\\log{(\phi_{k-1}/\phi_k)}\end{matrix} \right]\\
a(\eta) &= -\log{\frac{\phi_i}{\phi_k}}\\
b(y) &= 1
\end{align}
$$
于是对于每一个 $\eta_i$ 有链接函数：
$$
\eta_i = \log{(\frac{\phi_i}{\phi_k})}
$$
为了简单计算，我们给出定义 $\eta_i = \log{(\phi_k/\phi_k)} = 0$ 。且对链接函数取反函数然后推导出响应函数，就得到了下面的等式：
$$
\begin{align}
e^{\eta_i} & = \frac{\phi_i}{\phi_k}\\
\phi_{k}e^{\eta_i} &= \phi_i\\
\phi_{k}\sum^k_{i=1}e^{\eta_i}&=\sum^k_{i=1}\phi_i = 1
\end{align}
$$
这样得到 $\phi_k = 1/\sum^k_{i=1}e^{\eta_i}$ ，然后我们我们回代入 $e^{\eta_i} = \frac{\phi_i}{\phi_k}$ ，

得到相应函数：
$$
\phi_i = \frac{e^{\eta_i}}{\sum^k_{i=1}e^{\eta_i}}
$$
上面这个函数从 $\eta$ 映射到了$\phi$ ，称为 Softmax函数。通过假设3，我们有了 $\eta_i =\theta_i^Tx $ ，其中的$\theta_1,\theta_2, \dots ,\theta_{k-1} \in \mathbb{R}^{n+1}$ 就是参数了。我们这里还是定义 $\theta_k=0$ ，这样就有 $\eta_k = \theta_k^T x = 0$ ，与上文相呼应。

因此，我们有了模型：
$$
\begin{align}
p(y=i|x;\theta) &=\phi_i\\
&=\frac{e^{\eta_i}}{\sum^k_{i=1}e^{\eta_i}}\\
&=\frac{e^{\theta_i^Tx}}{\sum^k_{i=1}e^{\theta_i^Tx}}
\end{align}
$$
于是，我们的假设函数是：
$$
\begin{align}
h_\theta(x)&=E[T(y)|x;\theta]\\
&=E\left[\begin{array}{c|c}1\{y = 1\} &\\1\{y = 2\}\\1\{y = 3\}& x;\theta\\ \vdots\\1\{y = k-1\}\end{array}\right]\\
&=E\left[\begin{array}{c}\phi_1&\\\phi_2\\\phi_3\\ \vdots\\\phi_{k-1}\end{array}\right]\\
&=E\left[\begin{array}{c}
\frac{\rm{exp}^{\theta_1^Tx}}{\sum^k_{j=1}\rm{exp}^{\theta_j^Tx}}&\\
\frac{\rm{exp}^{\theta_2^Tx}}{\sum^k_{j=1}\rm{exp}^{\theta_j^Tx}}\\
\frac{\rm{exp}^{\theta_3^Tx}}{\sum^k_{j=1}\rm{exp}^{\theta_j^Tx}}\\\vdots\\
\frac{\rm{exp}^{\theta_{k-1}^Tx}}{\sum^k_{j=1}\rm{exp}^{\theta_j^Tx}}\end{array}\right]
\end{align}
$$
然后，对于一个训练集来说，我们为了求得 $\pmb{\theta}$ ，写出似然函数的对数：
$$
\begin{align}
\ell(\pmb{\theta}) &= \sum^m_{i=1}\log{p(y^{(i)}|x^{(i)};\theta)}\\
&=\sum^m_{i=1}\log{\prod^k_{l=1}\left(\frac{\rm{exp}^{\theta_{l}^Tx^{(i)}}}{\sum^k_{j=1}\rm{exp}^{\theta_j^Tx^{(i)}}}\right)^{1\{y^{(i)}=l\}}}
\end{align}
$$
然后我们可以通过梯度上升法或者牛顿法为求：
$$
\DeclareMathOperator*{\argmax}{arg\,max} 
\pmb{\theta} = \mathop{\argmax}_{}{\ell(\theta)}
$$
