# my-deep-learning
The way to my Deep Learning  
[Numpy 中文文档](https://www.numpy.org.cn/article/basics/numpy_matrices_vectors.html)
### 写在前面
* `Ctrl+/` **可快速注释单行或多行**
* 当前单元格内容显示/不显示行号（命令模式下）：`L`
* 当前单元格下方创建单元格（命令模式下）：`A`
* 当前单元格上方创建单元格 (命令模式下)：`B`
* 删除单元格：连续两次按 `D`
* `Esc+F` 在代码中查找、替换，(忽略输出)
* 当前单元格MarkDown模式和Code模式切换（命令模式下）：m到c为`Y`, c到m为`M`
* 快速跳转到首个cell：`Ctrl Home`
* 快速跳转到最后一个cell：`Ctrl End`
***
### Numpy创建数组
### 快速创建N维数组

### Numpy创建随机数组`np.random`
#### 均匀分布
> `np.random.rand(10, 10)`创建指定形状(示例为10行10列)的数组(范围在0至1之间)
>
> `np.random.uniform(0, 100)`创建指定范围内的一个数
>
> `np.random.randint(0, 100)` 创建指定范围内的一个整数
***
#### 正态分布
* 给定均值/标准差/维度的正态分布`np.random.normal(1.75, 0.1, (2, 3))`


## [线性代数](https://www.numpy.org.cn/user_guide/quickstart_tutorial/linear_algebra.html)
构造与矩阵对应的numpy数组
$  \left[
  \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 
  \end{matrix} 
  \right] $
我们会这样：
> `A = np.array([[1,2,3],[4,5,6]])`

向量只是具有单列的数组。 例如，构建向量
$  \left[
  \begin{matrix}
   2 \\
   1 \\
   3
  \end{matrix} 
  \right] $
我们这样：
> `v = np.array([[2],[1],[3]])`

更方便的方法是转置相应的行向量。 例如，为了使上面的矢量，我们可以改为转置行向量
$  \left[
  \begin{matrix}
   2 &  1& 3
  \end{matrix} 
  \right] $
也就是：
> `v = np.transpose(np.array([[2,1,3]]))`
***


### 矢量化
>$$ h_w(x)= \sum_{j=0}^{n}w_j x_j\\ = w^T x $$

# 多元线性回归

> $$  Y = W^T\cdot  X = w_0 x_0 + w_1 x_1 + w_2 x_2 +...+w_n x_n$$

**正规方程**给出$W$的解析解为：
> $$ W = (X^T X)^{-1}X^T Y  $$

注：`np.linalg.pinv(a)`将给出矩阵a的伪逆。   [更多](https://blog.csdn.net/gilzhy/article/details/8694715)

###  正规方程法

> $$ Y = W^T \cdot  X = w_0 x_0 + w_1 x_1 + w_2 x_2 +...+w_n x_n \tag{1}$$
其中
$$ W = \left[
  \begin{matrix}
   w_0^{(i)} \\
   w_1^{(i)} \\
   w_2^{(i)}\\
   ... \\
   w_n^{(i)}
  \end{matrix} 
  \right] \qquad 
X = 
\left[
  \begin{matrix}
   x_0^{(i)} \\
   x_1^{(i)} \\
   x_2^{(i)} \\
   ... \\
   x_n^{(i)}
  \end{matrix} 
  \right] \tag{2}
$$
*i.e.*对于m个样本中的某一个来说，有：
$$  \left[
  \begin{matrix}
   w_0^{(i)} &
   w_1^{(i)} &
   w_2^{(i)} &
   ... &
   w_n^{(i)}
  \end{matrix} 
  \right] \cdot 
  \left[
  \begin{matrix}
   x_0^{(i)} \\
   x_1^{(i)} \\
   x_2^{(i)}\\
   ...\\
   x_n^{(i)}
  \end{matrix} 
  \right] = y^{(i)} \tag{3}  
$$
  $$(i = 1,2,3,...,m)$$
>这里对于线性拟合，我们只需要求两个参数：$w_0 和 w_1$，而m，即训练样本数。
>
>所以本处应是：
$$
\left[
  \begin{matrix}
   w_0^{(i)} &
   w_1^{(i)} 
  \end{matrix} 
  \right] \cdot 
\left[
  \begin{matrix}
   x_0^{(i)} \\
   x_1^{(i)} 
  \end{matrix} 
  \right] =  y^{(i)}  \tag{4}
$$
*i.e.*
$$
\left[
  \begin{matrix}
   w_0^{(i)} &
   w_1^{(i)} 
  \end{matrix} 
  \right] \cdot 
\left[
  \begin{matrix}
   1 \\
   x_1^{(i)} 
  \end{matrix} 
  \right] =  y^{(i)}  
$$

***
而实际上在算法求解时，用到的矩阵$X$并不是这里的公式(3),而是公式(3)的转置，一个$m\times n$ 维度的矩阵：

>$$
X = 
\left[
  \begin{matrix}
   1 & x_1^{(1)}& x_2^{(1)}& ...  & x_n^{(1)} \\
   1 & x_1^{(2)}& x_2^{(2)}& ...  & x_n^{(2)} \\
\vdots& \vdots  & \vdots   &\ddots& \vdots  \\
   1 & x_1^{(m)}& x_2^{(m)}& ...  & x_n^{(m)}
  \end{matrix} \right]
  = \left[
  \begin{matrix}
   1 & (x^{(1)})^{T}\\
   1 & (x^{(2)})^{T}\\
   1 & \vdots \\
   1 & (x^{(m)})^{T}
  \end{matrix} \right]
$$
后来发现这种表达和西瓜书上一致。

#### 开始求解
[Python之Numpy数组拼接，组合，连接](https://www.cnblogs.com/huangshiyu13/p/6672828.html)


### 梯度下降法(Gradient Descent)
**Linear Regression Model**
>
> 假设函数：
$$h_\theta (x) = \theta_0 x_0^{(i)}+ \theta_1 x_1^{(i)} + ... +\theta_n x_n^{(i)} 
= \theta^T \cdot \left[
  \begin{matrix}
   x_0^{(i)} \\
   x_1^{(i)} \\
   x_2^{(i)} \\
   ... \\
   x_n^{(i)}
  \end{matrix} 
  \right]\\
= \left[
  \begin{matrix}
   x^{(1)} \\
   x^{(2)} \\
   \vdots \\
   x^{(m)}
  \end{matrix} 
  \right] \cdot \theta
  = \left[
  \begin{matrix}
   1 & x_1^{(1)}& x_2^{(1)}& ...  & x_n^{(1)} \\
   1 & x_1^{(2)}& x_2^{(2)}& ...  & x_n^{(2)} \\
\vdots& \vdots  & \vdots   &\ddots& \vdots  \\
   1 & x_1^{(m)}& x_2^{(m)}& ...  & x_n^{(m)}
  \end{matrix} \right] \cdot \theta =X \cdot \theta\\
\theta^T  = (\theta_0 \quad\theta_1\quad...\quad\theta_n )\qquad
i = 1,2,...,m
$$

> 代价函数：$$ J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2 $$

目标是找到代价函数的极小值，由于在函数处于极值时，函数的梯度为0，
所以我们让代价函数的梯度朝着0的方向迭代即可找到极值。
代价函数的梯度表示为：
>\begin{align}
\nabla_\theta J(\theta) &= \nabla_\theta\frac{1}{2m}\sum_{i=1}^{m}(h_\theta(x^{(i)})-y^{(i)})^2\\
&= \nabla_\theta\frac{1}{2m}\sum_{i=1}^{m}(\theta_0 x_0^{(i)}+\theta_1 x_1^{(i)}+...+\theta_n x_n^{(i)}-y^{(i)})^2\\
\end{align}
其中：    $\nabla_\theta = (\frac{\partial}{\partial\theta_0}\quad\frac{\partial}{\partial\theta_1}\quad...\quad\frac{\partial}{\partial\theta_n})$

于是：
>$$
\frac{\partial}{\partial\theta_0}J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{i})x_0^{(i)}\\
\frac{\partial}{\partial\theta_1}J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{i})x_1^{(i)}\\
\vdots \\
\frac{\partial}{\partial\theta_n}J(\theta) = \frac{1}{m} \sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{i})x_n^{(i)}
$$

迭代公式
>$$
\theta_0 \leftarrow \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{i})x_0^{(i)}\\
\theta_1 \leftarrow \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{i})x_1^{(i)}\\
\vdots \\
\theta_n \leftarrow \theta_n - \alpha \frac{1}{m} \sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{i})x_n^{(i)}
$$

我们设:
>$$T_j = \sum_{i=1}^{m}(h_\theta (x^{(i)})-y^{i})x_j^{(i)} = (X\cdot\theta-y)^T \cdot X_j \\
j = 0,1,...,n$$

于是：
>$$\theta_j \leftarrow \theta_j - \alpha \frac{1}{m} T_j $$

# 持续更新中
