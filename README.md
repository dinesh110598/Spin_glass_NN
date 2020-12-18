# Spin_glass_NN

## Brief summary
In this we take Edwards Anderson model in a *triangular* 2D lattice with nearest neighbour couplings sampled from a bimodal +/-1 distribtion. The probability of +1 against -1 in the bimodal distribution is denoted by *p*. In this scenario, we use a convolutional neural network (CNN) to classify between low temperature samples for different values of p between 0.5 and 1. In this range of values of *p*, the low temperature states are expected to be frustrated states with zero magnetizations. Hence, we can rule out the possibility of using magnetization as a useful "order parameter" in case we're attempting this classification manually. One needs to use 2-point (or higher) correlation functions for this purpose. This provides an evidence that a CNN is capable of learning statistics more complex than the Dense network learning to classify phases of a ferromagnetic Ising model in Ref (1). Ref (1) also provides a similar example of classifying low and high temperature phases of square ice model in which we cannot simply use magnetization as well. Moreover, the ground state of the square ice model is highly degenerate as is in any kind of spin glass state.

However, Ref (1) and similar papers in literature deal only with classifying between states of different temperatures, a scalar quantity that directly features in the underlying Boltzmann distribution: exp(-H/T). On the other hand, *p* doesn't appear explicitly in the Boltzmann distribution corresponding to our problem but is a stochastic hyperparameter, which potentially renders the system to have a **manifold** of degenerate ground states. So our problem tests the capability of convolutional neural networks one step further. It's emphasized that we're not aware of or proposing a phase transition in the system we're studying contrary to what Ref (1) and similar papers study.

Please open the main.ipynb file in Google colab with the interactive button at the top and then change runtime to "GPU", follow the instructions to reproduce the results. If the "blob" takes too long to load on github, follow this [link](https://colab.research.google.com/github/dinesh110598/Spin_glass_NN/blob/master/main.ipynb). To run it on a local machine, fork this repo locally. Note that a GPU with CUDA support is necessary to run on the local machine. In the software side, Python 3 with numpy, matplotlib, tensorflow 2.x and numba are required.

## References:
1. J Carrasquilla and R Melko, Machine Learning Phases of Matter, https://www.nature.com/articles/nphys4035
