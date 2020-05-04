## TextRank

[Source](../rank_words.py)

*此部分对应论文 §3.2.1 Word Extracting*

### PageRank 算法

记 $V=\{V_1,V_2,\cdots,V_|V|\}$ 为所有页面的集合。则

$$S(V_i):=(1-d)+d\cdot\sum_{V_j\in\operatorname{In}(V_i)}\frac1{\big|\operatorname{Out}(V_j)\big|}S(V_j)$$

### TextRank 算法

对称矩阵

$$\bm W=\begin{pmatrix}
w_{11}&w_{12}&\cdots&w_{1|V|}\\
w_{21}&w_{22}&\cdots&w_{2|V|}\\
\vdots&\vdots&&\vdots\\
w_{|V|1}&w_{|V|2}&\cdots&w_{|V||V|}
\end{pmatrix}$$

表示词 $V_i$ 和词 $V_j$ 之间的关系大小。这里直接采取共现矩阵。

$$S(V_i):=(1-d)+d\cdot\sum_{V_j\in\operatorname{In}(V_i)}\frac{w_{ij}}{\displaystyle\sum_{V_k\in\operatorname{Out}(V_j)}w_{jk}}S(V_j)$$


注意到共现矩阵是对称的，因此 $\operatorname{In}(V_i)\equiv\operatorname{Out}(V_i)$。同时若 $\bm W$ 中无关位置的元素取 $0$，则可写作：

$$S(V_i):=(1-d)+d\cdot\sum_{j=1}^{|V|}\frac{w_{ij}}{\displaystyle\sum_{k=1}^{|V|}w_{jk}}S(V_j)$$

可以对矩阵 $\bm W$ 进行预处理（正规化），即计算对称矩阵 $\bm R=(r_{ji})_{|V|\times|V|}$，且
$$r_{ji}=\frac{w_{ij}}{\displaystyle\sum_{k=1}^{|V|}w_{jk}}$$

那么可写作

$$S(V_i):=(1-d)+d\cdot\sum_{j=1}^{|V|}r_{ji}S(V_j)$$
