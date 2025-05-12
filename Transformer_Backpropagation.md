# Backpropagation in Transformer Architectures: A Detailed Mathematical Exposition

## I. Foundations of Backpropagation in Neural Networks

### A. The Chain Rule: The Mathematical Engine of Learning

The chain rule allows us to compute the derivative of a composite function.

If \( L = f(y_1, y_2, ..., y_m) \) and each \( y_i = g_i(x_1, ..., x_n) \), then:
\[ \frac{\partial L}{\partial x_j} = \sum_i \frac{\partial L}{\partial y_i} \cdot \frac{\partial y_i}{\partial x_j} \]

This rule is applied layer by layer, iterating backwards from the output.

Upstream Gradient: \( \frac{\partial L}{\partial y} \)

Local Gradient: \( \frac{\partial y}{\partial x} \)

Total Gradient: \( \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} \)

In neural networks, we compute:
\[
\frac{dL}{dW_1} = \frac{dL}{df_k} \cdot \frac{df_k}{df_{k-1}} \cdots \frac{df_2}{df_1} \cdot \frac{df_1}{dW_1}
\]

... (content truncated for brevity in this code snippet)
