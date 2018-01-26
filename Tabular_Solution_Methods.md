# Tabular Solution Methods

这一部分主要介绍那些状态空间和动作空间足够小的问题，小到我们可以通过数组或表格的形式来表达Value Function.

主要介绍

* Bandit problem 老虎机问题
* Markov decision process 马尔科夫决策过程
* Bellman equation 贝尔曼公式
* Dynamic programming 动态规划
  * 数学原理很强，但是model based, 我们需要对模型非常了解
* Monte Carlo methods
  * model-free, 但是不适用于迭代计算
* temporal-difference learning
  * model-free, 适用于迭代计算，但是分析起来比较复杂

这些方法的效率和收敛速度也是不同的。

* Dynamic programming, Monte Carlo, temporal-difference 混合，发挥各自的优势。

## Chapter 2 Multi-armed Bandits 多臂老虎机

**Evaluative feedback**, 告诉我们采取哪个动作有什么好处，以来于实际做了什么动作

**Instructive feedback**, 监督学习，不管我们实际做了什么，他都会告诉我们正确的动作

通过一个多臂老虎机的例子，我们可以比较**Evaluative feedback**和**Instructive feedback**的区别和如何将二者结合起来。

### 2.1 A k-armed Bandit Problem

问题描述: 重复的在k个不同的动作中选择一个动作，每选择一个动作，你都会收到一个服从依赖于你选择动作的一个固定概率分布的奖励。你的目标是经过1000轮选择后，使你获取的奖励最大化。

**Exploitation**, 使用greedy action，可以使这一个动作的收益最大化

**Exploration**, 随机选择动作，更新对该动作获得的奖励估计，更新估计后，再利用**Exploitation**,可以使长期收益最大化

使**Exploitation**和**Exploration**二者平衡是强化学习中的一个独特挑战。

### 2.2 Action-value Methods

$$
Q_{t}(a)=\frac{\text{sum of rewards when a taken prior to t}}{\text{number of times a taken prior to t}}\\=\frac{\sum_{i=1}^{t-1}R_{i}\cdot}{\sum_{i=1}^{t-1}}
$$

随着分母的增大，$Q_{t}(a)$ 会收敛到其真值 $q_{\star}(a)$, 这中方法为 sample-average method

初始化: $Q_{1}(a) = 0$.

**Greedy Exploitation**

$$
A_{t} = \arg\max_{a}Q_{t}(a)
$$

**$\epsilon$-greedy**, 每经过一段时间都会随机的选取其他的action, 而不是 greedy action. 这样可以探索其他的action是否比现在的greedy action还有更高的reward. 优点是可以让所有的action都收敛到其真值$q_{\star}(a)$.

从另一层面，告诉我们，选择到最优action的几率会收敛到大于$1-\epsilon$.

### 2.3 The 10-armed Testbed

$epsilon$-greedy 不一定比单纯的greedy方法好，这个要具体问题具体分析。

* Reward方差为0, stationary问题时，greedy要好于$epsilon$-greedy
* Reward方差大于0时或nonstatinary问题，选择$epsilon$-greedy会得到更多的reward. 因为探索会有可能找到更高reward的action.

**Stationary** 就是每个action对应的reward永远不变

**Nonstationary** 每个action对应的reward会变化

### 2.4 Incremental Implementation

$$
Q_{n+1}=Q_{n}+\frac{1}{n}(R_{n}-Q_{n})
$$

```
Initialize, for a = 1 to k:
  Q(a) = 0
  N(a) = 0

Repeat forever:
A = argmaxQ(a)           with probility 1-epsilon
A = a random action      with probility epsilon  
R = bandit(A)
N(A) = N(A) + 1
Q(A) = Q(A) + (R - Q(A)) / N(A)
```

### 2.5 Tracking a Nonstationary problem

强化学习问题一般都是Nonstationary的，在这种情况下，最近的reward权重应该大于历史奖励。常用的方法是将步长设置为常数。

$$
Q_{n+1}=Q_{n}+\alpha(R_{n}-Q_{n})
$$

### 2.6 Optimistic Initial Values

$Q_{1}$ 的初始值不是0，而是设置为一个较大的数，这个数可以比所有action对应的reward都大。这样的好处是，

我们使用greedy会选择到$Q_{1}(a_{1})$, 用实际的reward更新后，Q值变小了，而这时其他的action对应的Q值还是初始的大值，这样下一次greedy就会选择到其他的action. 这样就实现了便利所有的action的目的，实现了对所有action的探索。这种机制我们称之为**鼓励探索**。

### 2.7 Upper-Confidence-Bound Action Selection

$$
A_{t} = \arg\max_{a}\left[Q_{t}(a)+c\sqrt{\frac{\ln{t}}{N_{t}(a)}}\right]
$$

这种方法虽然精度高，但是更负责，尤其是处理 nonstationary 问题和大状态空间问题，因此在实际工程中并不实用。

### 2.8 Gradient Bandit Algorithms

$H_{t}(a)$, action的喜好数值值，这个值越大，代表将来选取a的几率也越大。

根据soft-max distribution

$$
Pr\{A_{t}=a\}=\frac{e^{H_{t}(a)}}{\sum_{b=1}^{k}e^{H_{t}(b)}}=\pi_{t}(a)
$$

$\pi_{t}(a)$, t时刻选取action a的概率。

**stochastic gradient ascent**

$$
H_{t+1}(A_{t})=H_{t}(A_{t})+\alpha(R_{t}-\bar{R_{t}})(1-\pi_{t}(A_{t})), \text{   and} \\
H_{t+1}(a)=H_{t}(a)-\alpha(R_{t}-\bar{R_{t}})\pi_{t}(a), \text{ for all }\alpha \neq A_{t}
$$

对于stationary问题
$$
\bar{R_{t}}=\bar{R_{t-1}}+\frac{R_{t-1}-\bar{R}_{t-1}}{t-1}
$$

对于nonstationary问题

$$
\bar{R_{t}}=\bar{R_{t-1}}+\alpha(R_{t-1}-\bar{R}_{t-1})
$$
