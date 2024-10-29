# MLP-KAN
Kolmogorov–Arnold Networks with modified activation (using MLP + positional encoding to represent the activation). The code utilizes `torch.vmap` to accelerate and simplify the process.

## Experiment

Running the following code for quick experiment:

```bash
python experiment.py
```

## Example usage

```python
from kan_layer import KANLayer

model = nn.Sequential(
        KANLayer(2, 5),
        KANLayer(5, 1)
    )

x = torch.randn(16, 2)
y = model(x)
# y.shape = (16, 1)
```

## Visualization

I experimented with a simple objective function:

$$f(x,y)=\exp(\sin(\pi x) + y^2)$$

```python
def target_fn(input):
    # f(x,y)=exp(sin(pi * x) + y^2)
    if len(input.shape) == 1:
        x, y = input
    else:
        x, y = input[:, 0], input[:, 1]
    return torch.exp(torch.sin(torch.pi * x) + y**2)
```

The first experiment set the network as:

```python
dims = [2, 5, 1]
model = nn.Sequential(
    KANLayer(dims[0], dims[1]),
    KANLayer(dims[1], dims[2])
)
```

After training on this, the activation function did learn the $\sin(\pi x)$ and $x^2$ functions:

![](./images/layer_0.png)

The exponential function is also been learned for the second layer:

![](./images/layer_1.png)

For better interpretability, we can set the network as:

```python
dims = [2, 1, 1]
model = nn.Sequential(
    KANLayer(dims[0], dims[1]),
    KANLayer(dims[1], dims[2])
)
```

Both the first layer and the second layer learning exactly the target function:

![](./images/layer_0_inter.png)

Second layer learning the exponential function:

![](./images/layer_1_inter.png)

# Linear Interpolation Version

```python
from kan_layer import KANInterpoLayer

model = nn.Sequential(
        KANInterpoLayer(2, 5),
        KANInterpoLayer(5, 1)
    )

x = torch.randn(16, 2)
y = model(x)
# y.shape = (16, 1)
```

The result shows similar performance. However, this version is harder to train. I guess it is because each parameter only affect the behavior locally, making it harder to cross local minima, or zero-gradient points. Adding `smooth_penalty` may help.

![](./images/layer_0_interpolation.png)

![](./images/layer_1_interpolation.png)

![](./images/loss_interpolation.png)
