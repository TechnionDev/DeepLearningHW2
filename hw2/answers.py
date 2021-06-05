r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd=5
    lr=0.05
    # reg=0.01
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd = 0.2
    lr_vanilla = 0.045
    lr_momentum = 0.003
    lr_rmsprop = 0.00016
    reg = 0.005
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    wstd = 0.1
    lr = 0.0023
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

1. 

"""

part3_q2 = r"""
**Your answer:**


1. The depth with best results was L4. This is in our opinion, because is was complex enough to express a good
approximation of reality but not too complex to suffer from vanishing gradients.
2. The network wasn't trainable for L8, L16 (for 32 features. For 64 features it was just L16). A way is to use skip
connections which will carry information that wouldn't vanish as easily. Another way to solve it would be to add partial
loss functions to different parts of the network. Essentially creating skip connections for the loss functions and
shortening the distance between them and the starting layers.

"""

part3_q3 = r"""
L4 and L8 produced similar results for K32 but L4 was much faster to train, thus, L4 is preferable in the experiment.

That being said, I think that for a different seed/more data/more epochs, L8 can potentially outperform L4 because it
seems that its graph is monotonically increasing and with an average of better results compared to L4.


Both in 1.1 and 1.2, increasing the size of the network too much (either width or depth), has negative impact on the
results. There is s sweet spot where the model provides the best results. 

 

"""

part3_q4 = r"""
**Your answer:**
Again we see a sweet spot (for L2) which produces the best results.
We can also see the volatility of simple network (L1). And for L4, the network becomes too deep the we get vanishing
gradients.
It also seems that there was an anomaly in the test **data**. It is likely the test rather than the model itself because
the dip happens for for L1 and L3 which are inherently different models.

"""

part3_q5 = r"""
**Your answer:**
In this experiment we use a resnet which allows us to train much deeper models without suffering from vanishing
gradients. We train networks of depth 32 which we couldn't do before (even L16 failed).

Compared to 1.1 and 1.3, there's a regression in the accuracy, most likely because of incompatibility of the hyper
parameters. These models were much slower to train which can be mitigated by using residual bottleneck blocks.

Also, like previous (and everything in life), moderation is key. Too much or too little, produces bad results. 

"""

part3_q6 = r"""
**Your answer:**
In experiment 2, the results were marginally improved even for relatively poor choice of hyper parameters. We can see in
the graphs that the potential is greater than in the experiments of 1.X.

 - It is possible that a better choice for `pool_every` (lower than 12) would greatly improve the results for the models
with L3, L6, L9. 
 - `hidden_dims` could also be tweaked (more layer and bigger layers) could also improve results. 
 - `lr` - modifying this value could also improve our model. We can see volatility which might indicate too high a value.
 - `reg` - changing the regularization might improve the performance of the model. 

"""
# ==============
