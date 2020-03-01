# Linear Regression

1. Create an array:

```py
n = 100 # number of rows
k = 2 # number of columns
torch.ones(n, k) # -> [[1.,1.], [1.,1.], ...]
```

[torch.ones()](https://pytorch.org/docs/stable/torch.html#torch.ones)

2. Set the first `x` to a random value (`y = a x0 + b x2`, `x2` will be `1`).

```py
x[:,0].uniform_(-1.,1)
```

[torch.Tensor.uniform_()](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.uniform_)

> When PyTorch has a method ending in `_`, it means it will replace the value in-place.

3. Create a tensor to work as the coefficients.

```py
a = tensor(3.,2)
```

[fastai.torch_core.tensor()](https://docs.fast.ai/torch_core.html#tensor): Helper to create tensor from arguments

[torch.tensor()](https://pytorch.org/docs/stable/torch.html#torch.tensor): PyTorch's actual method

4. Dot product the tensor and the `x` values to find a `y`.

```py
# x@a means the dot product of the 2 matrices
# torch.rand(n) adds noise to the result
y = x@a + torch.rand(n) 
```

5. Plot the chart using `pyplot`

```py
plt.scatter(x[:,0], y)
```

6. Find an `a` parameter that minimizes the error between the prediction and `x@a`. For most regression problems, the most common **loss function** or **error function** is the **mean squared error**

```py
# @return tensor
def mse(y_hat, y): return ((y_hat-y)**2).mean()

# Supposing the params are -1 and 1
a = tensor(-1.,1)

y_hat = x@a
mse(y_hat, y)

plt.scatter(x[:,0],y)
plt.scatter(x[:,0],y_hat)
```

We have specified a **model** (linear regression), an **evaluation criteria** or **loss function** (mean squared error). Next we define an **optimization**: how to find the best `a`, the best fitting linear regression?

7. Minimize the loss and find the best fitting linear regression

We want to minimize the loss. **Gradient descent** is an algorithm that minimizes functions: it starts with a set of initial values and iteratively moves to a set that minimizes the error.

![Steps of a gradient descent strategy](./img/shared/learning_rate_plot.png)

```py
# Create a parameter object using a random initial value
a = nn.Parameter(a)

# Update function to iteratively update `a`
def update():
    y_hat = x@a
    loss = mse(y, y_hat) # we are calculating over the whole set, in practice we use minibatches
    # if t % 10 == 0: print(loss)
    loss.backward() 
    with torch.no_grad():
        a.sub_(lr * a.grad)
        a.grad.zero_()

# Set learning rate
lr = 1e-1

# Iteratively run the update function
for t in range(100): update()

# Plot data and regression
plt.scatter(x[:,0],y)
plt.scatter(x[:,0],x@a)
```

**Concepts**

[Minibatch](./Glossary.md#Minibatch)

**Methods**

[torch.Tensor.backward()](https://pytorch.org/docs/stable/autograd.html#torch.Tensor.backward)
