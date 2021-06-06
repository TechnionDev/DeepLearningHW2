r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. Using the normal rules for tensor derivation, for each cell in the matrix, we derive by the whole of the matrix 
X, therefore, we would end up with a tensor of the shape of N x out_features x N x in_features.
2. if we store the matrix in a non-sparse way, as we probably would considering the computational cost of storing it
sparsely, we would need N x out_features X N x in_features x 4 = 137,438,953,472 bytes = 128TB of data
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
1. The graphs we see roughly match what we expected to see from the dropout results, where with no dropout, we see we 
overfit on the data, which results in poor results on the train set, with a medium dropout, we achieve our best results, 
as expected, where we don't overfit and also achieve better results on the test, and in high dropout, where we drop most
of our neurons, we achieve a highly random behavior, where it might achieve better or worse results, depending on the seed.
2. as described before, the high dropout achieved randomness in the graph, as can be seen in the jaggedness and non-monotonic
behaviour we see, and in the no dropout we overfit, as should be expected. 
"""

part2_q2 = r"""
**Your answer:**
It is possible, (it also has happened to us on part 3). this can happen in situations, where, in results where we err'd at, 
we had a big margin of error, and therefore, lost heavily on that, but in general, made fewer mistakes in the epoch, and therefore,
the test score increased. this can go on for a couple of epochs, where the gradient tries to minimize the score of those points
where it made an error on, and hurt a different subset of results in accordance, but still, improve on the whole.
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

1. The parameter count per layers (with bias) is  $C=out*(size^2*in+1)$. the param count for the example given, is
$C_1=2*256*(9*256+1)=1,180,160$ params, where in the bottleneck block, the result (with 2 3x3 conv in the middle) would be 
$C_2=64*(1*256+1)+64*(9*64+1)+256(1*64+1)=70,016$ params, and with 2 3x3 convolutions it would be 
$C_3=64*(1*256+1)+2*64*(9*64+1)+256(1*64+1)=106,944$ params, which is much *much* lower than the original residual block.
2. Obviously, the residual block would incur a higher performance cost with regards to FLOPS, seeing as we would perform
a higher-dimensional action, with a 3x3 convolution, against the residual bottleneck block, which would do the same 
action, on a lesser dimensional count.
3. This part is dependant on weather we execute 2 3x3 convolutions on the bottleneck block. if we do, the bottleneck block is 
able to combine spatially in the same manner as the regular old residual block,(we result in the same receptive field in the 
H x W dimensions), and by combining the channels in the intro to the bottleneck block, and redispersing them on the output,
we combine across feature maps, in a way where it is more difficult in the regular residual block.
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
