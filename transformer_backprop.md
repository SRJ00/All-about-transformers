# Backpropagation in Transformer Architectures: A Detailed Mathematical Exposition

## I. Foundations of Backpropagation in Neural Networks

### A. The Chain Rule: The Mathematical Engine of Learning

The chain rule allows us to compute the derivative of a composite function.

If $L = f(y_1, y_2, ..., y_m)$ and each $y_i = g_i(x_1, ..., x_n)$, then:
$\frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_j}$

This rule is applied layer by layer, iterating backwards from the output.

Upstream Gradient: $\frac{\partial L}{\partial y}$

Local Gradient: $\frac{\partial y}{\partial x}$

Total Gradient: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x}$

In neural networks, we compute:

$$
\frac{dL}{dW_1} = \frac{dL}{df_k} \cdot \frac{df_k}{df_{k-1}} \cdots \frac{df_2}{df_1} \cdot \frac{df_1}{dW_1}
$$

### B. Gradient Descent and Parameter Optimization

Gradient descent updates parameters via:
$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$

Adam optimizer, commonly used in Transformers, builds on gradient descent.

### C. Computation Graphs

A neural net can be viewed as a DAG:

* Nodes: Variables
* Edges: Functions

Backprop proceeds in reverse topological order, computing:
$\bar{v}_i = \sum_{j \in Ch(v_i)} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_i}$

This avoids redundant computations by caching forward values.

## II. The Transformer Architecture

### A. Encoder-Decoder Framework

* Encoder maps input sequence to latent space.
* Decoder generates output sequence using encoder output.
* Both encoder and decoder use $N = 6$ layers.

### B. Core Components

1. **Input Embedding**: $X_{emb} = WE[X_{ids}]$, $WE \in \mathbb{R}^{V \times d_{model}}$
2. **Positional Encoding**:
   $PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
   $PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
3. **Multi-Head Attention**:
   $head_i = \text{Attention}(Q_i, K_i, V_i)$
   $Q_i = XW_i^Q, K_i = XW_i^K, V_i = XW_i^V$
   $MHA(X) = \text{Concat}(head_1, ..., head_h)W^O$
4. **Scaled Dot-Product Attention**:
   $S = QK^T, \quad S_{scaled} = \frac{S}{\sqrt{d_k}}$
   $A = \text{softmax}(S_{scaled}), \quad Z = AV$
5. **Feed-Forward Network (FFN)**:
   $FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$
6. **Residual Connections & Layer Norm**:
   $\text{Output} = \text{LayerNorm}(X + \text{Dropout}(\text{Sublayer}(X)))$
7. **Output Layer**:
   $\text{Logits} = Z W_{out}, \quad P = \text{softmax}(\text{Logits})$

## III. Backpropagation Through Components

### A. Embedding Layer

$\frac{\partial L}{\partial W_E[v,d]} = \sum_{i=1}^T \mathbb{I}(X_{ids}[i] = v) \cdot \frac{\partial L}{\partial X_{emb}[i,d]}$

### B. Positional Encoding (fixed)

$\frac{\partial L}{\partial X_{emb}} = \frac{\partial L}{\partial X_{final}}$

### C. Scaled Dot-Product Attention

1. $\frac{\partial L}{\partial V} = A^T \cdot dZ$
2. $\frac{\partial L}{\partial A} = dZ \cdot V^T$
3. $\frac{\partial L}{\partial S_{scaled}}= A \odot (dA - \text{row-sum}(dA \odot A))$
4. $\frac{\partial L}{\partial S} = \frac{1}{\sqrt{d_k}} \cdot \frac{\partial L}{\partial S_{scaled}}$
5. $\frac{\partial L}{\partial K} = (\frac{\partial L}{\partial S})^T Q$
6. $\frac{\partial L}{\partial Q} = \frac{\partial L}{\partial S} K$

### D. Multi-Head Attention

1. Output Gradient: $dZ \rightarrow dW^O = Z^T dZ$
2. Split dZ into $d\text{head}_i$, apply SDPA backward for each.
3. Sum contributions to $\frac{\partial L}{\partial X}$ via projection paths.

### E. Feed-Forward Network

1. $dW_2 = A_1^T dY, \quad db_2 = \sum dY$
2. $dZ_1 = dA_1 \odot \text{ReLU}'(Z_1)$
3. $dW_1 = X^T dZ_1, \quad db_1 = \sum dZ_1$
4. $dX = dZ_1 W_1^T$

### F. Layer Normalization

Let $\hat{X} = \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}}$

1. $d\beta = dY, \quad d\gamma = dY \odot \hat{X}$
2. $dX = \frac{1}{\sqrt{\sigma^2 + \epsilon}} \left( d\hat{X} - \frac{1}{D} \sum d\hat{X} - \hat{X} \cdot \frac{1}{D} \sum (d\hat{X} \cdot \hat{X}) \right)$

### G. Residual Connection

$\frac{\partial L}{\partial X_{input}} = \frac{\partial L}{\partial Y}$

### H. Output Layer with Softmax & Cross-Entropy

If $y_t$ is true label:
$\frac{\partial L_t}{\partial \\ logits_t} = P_t - y_t$ ,

$\frac{\partial L}{\partial W_{out}} = Z^T d\text{logits} \quad \text{and} \quad \frac{\partial L}{\partial Z} = d\text{logits} W_{out}^T$

## IV. Full Transformer Layer Backward Flow

### Encoder Layer

1. Backprop LayerNorm2 $\rightarrow$ Add2 $\rightarrow$ FFN
2. Add gradients from FFN and LayerNorm1
3. Backprop Add1 and MHA

### Decoder Layer

1. Same as encoder +
2. Backprop Cross-Attention
3. Gradients flow from decoder to encoder via encoder-decoder attention

## V. Implementation Notes

* Use NumPy arrays with careful shape tracking.
* Cache forward activations.
* Perform numerical gradient checks.
* Memory-intensive models require attention to activation storage.

## VI. Conclusion

Transformers are modular architectures. Backpropagation through each part is tractable with matrix calculus and chain rule. Implementing it from scratch (e.g. in NumPy) deepens understanding and reveals practical insights.

---

Let me know if you want this markdown exported as a file or rendered for preview.
