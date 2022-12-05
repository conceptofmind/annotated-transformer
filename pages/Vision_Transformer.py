import streamlit as st

# Global Variables

language='python'

# Section for Title

st.title('Vision Transformer')

st.image(
    "https://media.giphy.com/media/lr9Xd6IkR8F0mAv0bL/giphy.gif",
    caption="Animated Vision Transformer",
    #width=400, # The actual size of most gifs on GIPHY are really small, and using the column-width parameter would make it weirdly big. So I would suggest adjusting the width manually!
)

# Section for Table of Contents

st.header('Table of contents')

st.markdown('''
- [Introduction](#Introduction)
    - [Paper Abstract](#PaperAbstract)
    - [Acknowledgment](#Acknowledgement)
- [Prerequisites](#Prerequisites)
    - [Installation](#Installs)
    - [Imports](#Imports)
    - [Configuration](#Configuration)
    - [Helper Functions](#Helpers)
- [Model Architecture](#Model)
    - [Pre-Normalization](#prenorm)
    - [Multilayer Perceptron](#feedforward)
    - [Attention Mechanism](#attention)
    - [Transformer Network](#transformer)
    - [Vision Transformer Model](#visiontransformer)
- [Training](#Training)
    - [Initialize Model](#InitializeModel)
    - [Image Augmentation](#ImageAugmentation)
    - [CIFAR10 Dataset](#Dataset)
    - [Dataloader](#Dataloader)
    - [Loss Function](#LossFunction)
    - [Optimizer](#Optimizer)
    - [Learning Rate Scheduler](#LRS)
    - [Train Step](#TrainStep)
    - [Validation Step](#ValidationStep)
    - [Train the Model](#TrainingLoop)
    - [Save Trained Model](#SaveModel)
    - [Load Saved Model](#LoadModel)
    - [Make Predictions](#MakePredictions)
- [References](#References)
- [Citations](#Citations) 
''')

# Section for Introduction

st.header('Introduction', anchor='Introduction')

st.subheader('Paper Abstract', anchor='PaperAbstract')

st.markdown('''"
While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
" - Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby''')

# Section for acknowledgement

st.subheader('Acknowledgement', anchor='Acknowledgement')

st.write('''

''')

# Section for Prereqs

st.header('Prerequisites', anchor='Prerequisites')

st.write('''It is expected that you at least have some basic working knowledge of Python and PyTorch or Haiku. 
Deep learning greatly benefits from GPU or TPU acceleration. 
You will want to have access to a machine with one or many accelerated devices. 
If you do not have access to a GPU or TPU you can still use a CPU, although training times will be significantly longer. 
 
You can check the version of CUDA from the command line with:

`nvcc --version`

Or check the devices available on your machine with:

`nvidia-smi` 
''')

# Installation section

st.subheader('Installation', anchor='Installs')

st.write("You will first need to install either DeepMind's Haiku and Google's Jax, or Facebook's PyTorch.")

installs_tab_1, installs_tab_2 = st.tabs(["PyTorch", "Haiku"])

with installs_tab_2:
    st.write('Install Haiku:')
    haiku_installs = '''
$ pip3 install -U jax jaxlib dm-haiku
    '''
    st.code(haiku_installs, language='bash')

with installs_tab_1:
    st.write('Install PyTorch:')
    pytorch_installs = '''
$ pip3 install -U torch torchvision torchaudio
    '''
    st.code(pytorch_installs, language='bash')

    st.write("Check if PyTorch was successfully installed from the command line with:")

    st.code('python3 -c "import torch; print(torch.__version__)"', language='bash')

# Imports section

st.subheader('Imports', anchor='Imports')

st.write("You will need to import the necessary libraries in your Python file or Jupyter Notebook.")

imports_tab_1, imports_tab_2 = st.tabs(["PyTorch", "Haiku"])

with imports_tab_2:
    haiku_imports = '''
from functools import partial

import haiku as hk
from haiku import PRNGSequence

import jax
from jax import random, nn
import jax.numpy as jnp

import optax

from einops import rearrange, repeat
    '''
    st.code(haiku_imports, language)

with imports_tab_1:
    pytorch_imports = '''
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
    '''
    st.code(pytorch_imports, language)

# Configuration for Global Variables

st.subheader('Configuration', anchor='Configuration')

st.write("""A configuration class for defining global variables to be used for training the model.
Each of these variables are explained in detail in the relevant sections below.
""")

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
class CFG:
    learning_rate = 0.001
    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
class CFG:
    learning_rate = 0.001
    policy = T.AutoAugmentPolicy.CIFAR10
    image_size = 224
    num_classes = 10
    batch_size = 4
    device = 'cuda'
    seed = 42
    '''
    st.code(pytorch, language)

# Helper functions section

st.subheader('Helper Functions', anchor='Helpers')

st.write('Define some basic helper functions for the Model.')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
def pair(t):
    return t if isinstance(t, tuple) else (t, t)
    '''
    st.code(haiku, language)

    st.write('Haiku does not possess an Identity Layer class so we will want to define one as well.')

    identity_layer_class = '''
class IdentityLayer(hk.Module):
    def __call__(self, x):
        x = hk.Sequential([])
        return x
    '''
    st.code(identity_layer_class, language)

with tab_1:
    pytorch = '''
def seed_environment(seed):
      torch.manual_seed(seed)

seed_environment(CFG.seed)  

def pair(t):
    return t if isinstance(t, tuple) else (t, t)
    '''
    st.code(pytorch, language)

# Section for Model

st.header('Model Architecture', anchor='Model')

st.subheader('Pre-Normalization', anchor='prenorm')

st.write('''
Layer normalisation explicitly controls the mean and variance of individual neural network activations

Next, the output reinforced by residual connections goes through a layer normalization layer. Layer normalization, 
similar to batch normalization is a way to reduce the “covariate shift” in neural networks allowing them to be trained 
faster and achieve better performance. Covariate shift refers to changes in the distribution of neural network 
activations (caused by changes in the data distribution), that transpires as the model goes through model training. 
Such changes in the distribution hurts consistency during model training and negatively impact the model. It was 
introduced in the paper, “Layer Normalization” by Ba et. al. (https://arxiv.org/pdf/1607.06450.pdf).

However, layer normalization computes mean and variance (i.e. the normalization terms) of the activations in such a way 
that, the normalization terms are the same for every hidden unit. In other words, layer normalization has a single mean 
and a variance value for all the hidden units in a layer. This is in contrast to batch normalization that maintains 
individual mean and variance values for each hidden unit in a layer.  Moreover, unlike batch normalization, layer 
normalization does not average over the samples in the batch, rather leave the averaging out and have different 
normalization terms for different inputs. By having a mean and variance per-sample, layer normalization gets rid of the 
dependency on the mini-batch size. For more details about this method, please refer the original paper.

''')

st.latex(r'''
\mu^{l} = {\frac 1 H} \displaystyle\sum_{i=1}^H a_i^l
''')

st.latex(r'''
\sigma^l = \sqrt{{\frac 1 H} \displaystyle\sum_{i=1}^H (a_i^l - \mu^l)^2}
''')

st.latex(r'''
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
''')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
LayerNorm = partial(hk.LayerNorm, create_scale=True, create_offset=False, axis=-1)
    
class PreNorm(hk.Module):
    def __init__(self, fn):
        super(PreNorm, self).__init__()
        self.norm = LayerNorm()
        self.fn = fn
    def __call__(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    '''
    st.code(pytorch, language)

st.write("""
This is a class implementation of a pre-normalization layer, which is a composite layer consisting of a layer 
normalization layer followed by another layer or function. The `__init__` method initializes the pre-normalization layer 
with a given input dimension, `dim`, and a function, `fn`, which represents the layer or function that will be applied 
after the normalization layer.

The `forward` method defines the forward pass of the pre-normalization layer, where the input, `x`, is first passed through 
a layer normalization layer with `dim` dimensions. The output of the normalization layer is then passed through the 
function `fn`, along with any additional keyword arguments provided in `kwargs`. The output of the `forward` method is the 
result of applying the `fn` function to the normalized input.

The pre-normalization layer allows the input to be normalized before it is passed to the `fn` function, which can help 
improve the performance and stability of the model. This is especially useful when the `fn` function is a non-linear 
function, such as an activation function, that can benefit from input normalization.
""")

st.write("""
`self.norm` is an instance of the `nn.LayerNorm` class from PyTorch. This class represents a layer normalization operation, 
which is a type of normalization that is applied to the inputs of a layer in a neural network. The `nn.LayerNorm` class 
takes a single argument `dim`, which specifies the dimensions of the input data that will be normalized. In this case, the 
value of `dim` is passed directly to the `nn.LayerNorm` class, so the dimensions of the input data will be the same as the 
value of `dim`.
""")



st.subheader('Multilayer Perceptron', anchor='feedforward')

st.write("""
We propose the Gaussian Error Linear Unit (GELU), a high-performing neuralnetwork activation function. 
The GELU activation function is xΦ(x), where Φ(x) the standard Gaussian cumulative distribution function. 
The GELU nonlinearity weights inputs by their value
""")
st.write("The Gaussian Error Linear Unit (GELU) is defined as:")
st.latex(r"""
\text{GELU}(x) = xP(X \leq x) = x\Phi(x) = x * {\frac 1 2} \big[1 = \text{erf}\big(x/\sqrt2\big)\big]
""")
st.write("GELU is approximated with if greater feedforward speed is worth the cost of exactness.:")
st.latex(r'''
\text{GELU}(x) = 0.5 * x * \bigg(1 + \tanh\bigg(\sqrt{\frac 2 \pi} * (x + 0.044715 * x^3)\bigg)\bigg)
''')
st.write("""
The GELU (Gaussian Error Linear Unit) function is a type of activation function used in neural networks. It is defined 
as a function of the input, `x`, as shown in the equation above. The GELU function is a smooth approximation of the 
rectified linear unit (ReLU) activation function and has been shown to improve the performance of neural networks in 
some cases. The GELU function outputs values in the range [0, 1], with input values close to 0 resulting in output 
values close to 0 and input values close to 1 resulting in output values close to 1. This allows the GELU function to 
retain information about the magnitude of the input, which can be useful for certain types of learning tasks.
""")

st.write("""
`nn.GELU()` is a PyTorch function that creates a GELU activation function. The GELU activation function is a 
differentiable function that takes as input a tensor with any shape and returns a tensor with the same shape. The 
function applies the GELU function elementwise to the input tensor, resulting in a tensor of the same shape with values 
in the range [0, 1]. This allows the GELU activation function to be used in the forward pass of a neural network, 
allowing the network to learn non-linear transformations of the input data.
""")

st.write('dropout')
st.write("""
During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a 
Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as 
described in the paper Improving neural networks by preventing co-adaptation of feature detectors .

Furthermore, the outputs are scaled by a factor of \frac{1}{1-p} 
  during training. This means that during evaluation the module simply computes an identity function.
p – probability of an element to be zeroed. Default: 0.5
""")

st.write("""
`nn.Dropout(dropout)` is a PyTorch function that creates a dropout layer with a given dropout rate. The dropout layer is 
a regularization technique that randomly sets a fraction of the input elements to 0 during the forward pass, with the 
fraction determined by the dropout rate. This has the effect of reducing the dependence of each output element on a 
specific subset of the input elements, making the model less susceptible to overfitting and improving generalization 
performance. The `dropout` argument determines the dropout rate, which is the fraction of input elements that will be set 
to 0. A dropout rate of 0 means that no elements will be dropped, while a dropout rate of 1 means that all elements 
will be dropped. The default value for the `dropout` argument is 0, which means that no elements will be dropped by the 
dropout layer.
""")

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
class MLP(hk.Module):
    def __init__(self, dim, hidden_dim):
        super(FeedForward, self).__init__()
        self.linear1 = hk.Linear(hidden_dim)
        self.linear2 = hk.Linear(dim)
    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.gelu(x)
        x = hk.dropout(hk.next_rng_key(), rate = 0.0, x = x)
        x = self.linear2(x)
        x = hk.dropout(hk.next_rng_key(), rate = 0.0, x = x)
        return x
    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    '''
    st.code(pytorch, language)

st.write("""
This is a class implementation of a multi-layer perceptron (MLP), a type of neural network. The `__init__` method 
initializes the MLP with a given input dimension, `dim`, and hidden dimension, `hidden_dim`, as well as a dropout rate, 
`dropout`, which is set to 0 by default.

The `forward` method defines the forward pass of the MLP, where the input, `x`, is passed through a series of linear layers 
followed by a GELU non-linearity and dropout regularization. The output of the forward pass is the result of passing 
the input through the defined sequence of layers.

The `__init__` method initializes the MLP by defining the sequence of layers that make up the network. 
The first layer is a linear layer with `dim` input dimensions and `hidden_dim` output dimensions. This layer is followed by 
a GELU activation function and a dropout layer with a dropout rate of `dropout`. The next layer is another linear layer 
with `hidden_dim` input dimensions and `dim` output dimensions, followed by another dropout layer with the same dropout rate. 
The sequence of layers is then stored in the `net` attribute of the MLP. This sequence of layers defines the architecture 
of the MLP and determines how the input data is transformed as it passes through the network.

The `nn.Sequential` class is a PyTorch class that allows a sequence of layers to be defined and treated as a single, 
composite layer. In this case, the `nn.Sequential` class is used to define a sequence of five layers: a linear layer, 
a GELU activation function, a dropout layer, another linear layer, and another dropout layer. This sequence of layers 
is then treated as a single, composite layer that can be used in the forward pass of the MLP.
""")

st.write("""
This code creates a neural network using PyTorch, which is a popular deep learning framework. The network consists of a 
sequence of five layers, which are defined using the `nn.Sequential` class. The first layer is a linear layer, which 
applies a linear transformation to the input data. The second layer is a GELU (Gaussian Error Linear Unit) activation 
layer, which applies the GELU nonlinearity to the output of the previous layer. The GELU nonlinearity is a smooth, 
monotonic function that has been shown to improve the performance of deep learning models. The third layer is a dropout 
layer, which randomly sets some of the output values to zero. This is a regularization technique that helps to prevent 
the model from overfitting to the training data. The fourth layer is another linear layer, which applies another linear 
transformation to the output of the previous layer. The fifth and final layer is another dropout layer, which again 
randomly sets some of the output values to zero. The resulting network takes an input vector of size `dim`, applies a
 series of linear and nonlinear transformations to it, and produces an output vector of the same size.
""")

st.subheader('Attention Mechanism', anchor='attention')

st.write("""
An attention function can be described as mapping a query and a set of key-value pairs to an output,
where the query, keys, values, and output are all vectors. The output is computed as a weighted sum
of the values, where the weight assigned to each value is computed by a compatibility function of the
query with the corresponding key.

This code defines a class called `Attention` which extends the `nn.Module` class from the PyTorch library. The `Attention` 
class is a neural network module for computing attention. It has several key components:

- `__init__`: the constructor function for the class, which initializes the various layers and submodules of the 
    network, such as a softmax layer for computing attention, dropout layers for regularization, and linear layers for 
    projecting the input tensor into different subspaces.

- `forward`: the forward propagation function, which takes an input tensor x and applies the various layers of the 
    network in sequence to produce the output. This includes computing dot products between the query, key, and value 
    tensors, applying softmax to the dot products to compute the attention weights, and then using these attention weights 
    to compute the weighted sum of the values.
""")

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
class Attention(hk.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_qkv = hk.Linear(output_size = inner_dim * 3, with_bias = False)
        self.to_out = hk.Linear(dim) if project_out else IdentityLayer()

    def __call__(self, x):
        qkv = self.to_qkv(x)
        qkv = jnp.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = jnp.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = nn.softmax(dots, axis = -1)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)

        out = hk.dropout(hk.next_rng_key(), rate = 0.0, x = out)
        return out
    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    '''
    st.code(pytorch, language)

st.write("""
The `Attention` class can be used as a building block for more complex neural networks that need to compute attention 
over some input. By specifying different values for the hyperparameters `dim`, `heads`, `dim_head`, and `dropout`, the behavior 
of the attention mechanism can be customized to suit different tasks and applications.
""")

st.write("""
The `__init__` function of the `Attention` class is the constructor function, which is called when a new instance of 
the class is created. It initializes the various layers and submodules of the attention network, such as the softmax 
layer, dropout layers, and linear layers. 
""")

st.code("""
def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
    super().__init__()
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.attend = nn.Softmax(dim = -1)
    self.dropout = nn.Dropout(dropout)

    self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

    self.to_out = nn.Sequential(
        nn.Linear(inner_dim, dim),
        nn.Dropout(dropout)
    ) if project_out else nn.Identity()
""", language='python')

st.write("""
Here is a detailed breakdown of what happens in each line of the `__init__` function:

1. The `super().__init__()` line calls the constructor of the `nn.Module` class, which is the base class for all neural 
    network modules in PyTorch. This initializes the `nn.Module` class with the `Attention` class as its child.

2. The `inner_dim` variable is set to the product of the `dim_head` and `heads` hyperparameters. This will be the size of the 
    inner subspaces that the input tensor is projected into by the `self.to_qkv` layer.

3. The `project_out` variable is set to `True` if the number of heads is not equal to 1 or the dimension of the head is not 
    equal to the original dimension of the input tensor. This will be used to determine whether the output tensor should be 
    projected back into the original space of the input tensor.

4. The `self.heads` attribute is set to the value of the heads hyperparameter. This specifies the number of heads in the 
    attention mechanism.

5. The `self.scale` attribute is set to the inverse square root of the dimension of the head tensor. This will be used to 
    scale the dot products of the query and key tensors.

6. The `self.attend` attribute is set to a new `nn.Softmax` layer, which will be used to compute the attention weights from 
    the dot products of the query and key tensors.

7. The `self.dropout` attribute is set to a new `nn.Dropout` layer, which will be used to apply dropout regularization to 
    the attention weights tensor.

8. The `self.to_qkv` attribute is set to a new linear layer, which will be used to project the input tensor into the 
    query, key, and value subspaces.

9. The `self.to_out` attribute is set to either a new `nn.Sequential` module containing a linear layer and a dropout layer, 
    or an `nn.Identity` layer depending on the value of the `project_out` variable. This will be used to project the output 
    tensor back into the original space of the input tensor if necessary.
""")

st.write("""
The `forward` function of the `Attention` class takes an input tensor `x` and applies the attention mechanism to it. 
""")

st.code("""
def forward(self, x):
    qkv = self.to_qkv(x).chunk(3, dim = -1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    attn = self.attend(dots)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)
""", language)

st.write("""
Here is a detailed breakdown of what happens in each line of the `forward` function:

1. The input tensor `x` is projected into three subspaces using a linear layer `self.to_qkv`. These subspaces correspond to 
    the query, key, and value tensors used in the attention mechanism. The resulting tensor is then split into three parts 
    along the last dimension, using the `chunk` method.

2. The query, key, and value tensors are rearranged using the `rearrange` function, which applies a specified reshaping 
    operation to a tensor. In this case, the reshaping operation is defined by the string `'b n (h d) -> b h n d'`, which 
    specifies that the tensors should be reshaped such that the batch and head dimensions are interleaved.

3. The query and key tensors are multiplied together using the `torch.matmul` function, and then scaled by the value 
    `self.scale`, which is the inverse square root of the dimensions of the head tensor. This produces a tensor of dot 
    products, which can be interpreted as the similarity between each query and key.

4. The dot products tensor is passed through the softmax function using the `self.attend` layer, which produces a tensor 
    of attention weights. These weights represent the importance of each value in the output.

5. The attention weights tensor is passed through the `self.dropout` layer, which applies dropout regularization to 
    prevent overfitting.

6. The attention weights tensor is multiplied by the value tensor, using the `torch.matmul` function, to compute the 
    weighted sum of the values. This is the output of the attention mechanism.

7. The output tensor is reshaped using the `rearrange` function and then passed through the `self.to_out layer`, which 
    projects it back into the original space of the input tensor. This is the final output of the `forward` function.
""")

st.subheader('Transformer Encoder', anchor='transformer')

#st.image('')

st.write("""
The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before
every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two
sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected 
feed-forward network. We employ a residual connection [11] around each of
the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is
LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer
itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding
layers, produce outputs of dimension dmodel = 512.
""")

st.write("""
The `TransformerEncoder` class extends the `nn.Module` class from the PyTorch library. It is a neural network module that 
implements a transformer encoder, which is a type of recurrent neural network that uses self-attention to compute a 
weighted sum of its inputs. It has several key components:

- `__init__`: the constructor function for the class, which initializes the various layers and submodules of the network, 
such as the `Attention` and `MLP` layers. It also creates a list of `PreNorm` layers, which are used to normalize the inputs 
to the attention and `MLP` layers.

- `forward`: the forward propagation function, which takes an input tensor `x` and applies the various layers of the network 
in sequence to produce the output. This includes applying the attention and MLP layers, adding the output of each layer 
to the input, and then returning the final result.

The `TransformerEncoder` class can be used as a building block for more complex neural networks that need to compute 
self-attention over some input. By specifying different values for the hyperparameters `dim`, `depth`, `heads`, `dim_head`, 
`mlp_dim`, and `dropout`, the behavior of the transformer encoder can be customized to suit different tasks and applications.
""")

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
class Transformer(hk.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super(Transformer, self).__init__()
        self.layers = []
        for _ in range(depth):
            self.layers.append([
                PreNorm(Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(MLP(dim, mlp_dim))
            ])
    def __call__(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
class TransformerEncoder(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, MLP(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    '''
    st.code(pytorch, language)

st.code("""
def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
    super().__init__()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
        self.layers.append(nn.ModuleList([
            PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
            PreNorm(dim, MLP(dim, mlp_dim, dropout = dropout))
        ]))
""", language)

st.write("""
The `__init__` function of the `TransformerEncoder` class is the constructor function, which is called when a new instance 
of the class is created. It initializes the various layers and submodules of the transformer encoder, such as the 
`Attention` and `MLP` layers. Here is a detailed breakdown of what happens in each line of the `__init__` function:

1. The `super().__init__()` line calls the constructor of the `nn.Module` class, which is the base class for all neural 
    network modules in PyTorch. This initializes the nn.Module class with the `TransformerEncoder` class as its child.

2. The `self.layers` attribute is set to a new `nn.ModuleList` object, which is a list of neural network modules. This list 
    will be used to store the `PreNorm` layers that normalize the inputs to the attention and MLP layers.

3. A `for` loop iterates over the range of the `depth` hyperparameter, which specifies the number of layers in the transformer 
    encoder. For each iteration of the loop, a new `PreNorm` layer is created for the attention and MLP layers, and then 
    appended to the `self.layers` list.

4. The `PreNorm` layers are created using the `dim` hyperparameter, which specifies the dimension of the input and output 
    tensors, and the `Attention` and `MLP` layers, which are initialized with the specified hyperparameters. The `PreNorm` layers 
    are used to normalize the inputs to the attention and MLP layers, which helps improve the stability and performance of 
    the transformer encoder.
""")

st.code("""
def forward(self, x):
    for attn, ff in self.layers:
        x = attn(x) + x
        x = ff(x) + x
    return x
""", language)

st.subheader('Vision Transformer Model', anchor='visiontransformer')

st.write("""
We split an image into fixed-size patches, linearly embed each of them,
add position embeddings, and feed the resulting sequence of vectors to a standard Transformer
encoder. In order to perform classification, we use the standard approach of adding an extra learnable
“classification token” to the sequence.
""")

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
class VitBase(hk.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64):
        super(VitBase, self).__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_height = patch_height
        self.patch_width = patch_width

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = hk.Linear(dim)

        self.pos_embedding = hk.get_parameter('pos_embedding', shape = [1, num_patches + 1, dim], init = jnp.zeros)
        self.cls_token = hk.get_parameter('cls_token', shape = [1, 1, dim], init = jnp.zeros)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.pool = pool

        self.mlp_head = hk.Sequential([
            LayerNorm(),
            hk.Linear(num_classes)
        ])

    def __call__(self, img):

        img = rearrange(img, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = self.patch_height, p2 = self.patch_width)

        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = jnp.concatenate([cls_tokens, x], axis = 1)
        x += self.pos_embedding[:, :(n + 1)]
        x = hk.dropout(hk.next_rng_key(), rate = 0.0, x = x)

        x = self.transformer(x)

        if self.pool == 'mean':
            x = jnp.mean(x, axis = 1)
        else:
            x = x[:, 0]

        x = self.mlp_head(x)

        return x
    '''
    st.code(haiku, language)

    st.write('Haiku requires ')

    haiku_transform = '''
def ViT(**kwargs):
    @hk.transform
    def inner(img):
        return VitBase(**kwargs)(img)
    return inner
    '''
    st.code(haiku_transform, language)

with tab_1:
    pytorch = '''
class ViT(nn.Module):
    def __init__(
        self, 
        *, 
        image_size, 
        patch_size, 
        num_classes, 
        dim, 
        depth, 
        heads, 
        mlp_dim, 
        pool = 'cls', 
        channels = 3, 
        dim_head = 64, 
        dropout = 0., 
        emb_dropout = 0.
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
    '''
    st.code(pytorch, language)

st.write("""
This is a class implementation of a vision transformer (ViT), a type of neural network that uses self-attention 
mechanisms to process visual data. The `__init__` method initializes the ViT with several hyperparameters, including the 
size of the input images, the patch size, the number of classes, the dimension of the hidden layers, the depth of the 
transformer encoder, the number of attention heads, the dimension of the MLP layers, the pooling method, the number of 
channels in the input images, and the dropout rate.

The `forward` method defines the forward pass of the ViT, where the input image is first split into patches and 
transformed into patch embeddings using a linear layer. The patch embeddings are then concatenated with a special 
"class" token and passed through a transformer encoder, which applies self-attention mechanisms to the input. The 
output of the transformer encoder is then either pooled using mean pooling or reduced to a single vector using the 
"class" token, depending on the `pool` parameter. The final output of the ViT is the result of passing the pooled or 
reduced vector through a linear layer and a layer normalization layer.

The ViT model is a flexible and powerful model that can be used for a wide range of computer vision tasks. It has the 
ability to process inputs of arbitrary size and to capture long-range dependencies in data, which makes it well-suited 
for many types of visual data. However, it also has a large number of hyperparameters, which can make it challenging to 
train and optimize.
""")

st.write("""
The `__init__` method initializes the ViT with several hyperparameters that determine the architecture and behavior of 
the model. The `image_size` parameter determines the size of the input images, which should be a tuple of the form 
`(image_height, image_width)`. The `patch_size` parameter determines the size of the patches into which the input images 
will be split, which should also be a tuple of the form `(patch_height, patch_width)`. The `num_classes` parameter 
determines the number of classes that the ViT will be trained to predict. The `dim` parameter determines the dimension of 
the hidden layers in the ViT.

The `depth` parameter determines the number of layers in the transformer encoder, which is the core component of the ViT 
that applies self-attention mechanisms to the input. The `heads` parameter determines the number of attention heads that 
will be used in the transformer encoder. The `mlp_dim` parameter determines the dimension of the MLP layers that are used 
in the transformer encoder. The `pool` parameter determines the pooling method that will be used to reduce the output of 
the transformer encoder, which can be either 'cls' (class token pooling) or 'mean' (mean pooling).

The `channels` parameter determines the number of channels in the input images, which should be 3 for color images and 1 
for grayscale images. The `dim_head` parameter determines the dimension of the attention heads used in the transformer 
encoder. The `dropout` parameter determines the dropout rate that will be used in the transformer encoder.

The `emb_dropout` parameter determines the dropout rate that will be used on the patch embeddings after they are 
concatenated with the "class" token.

After the hyperparameters are set, the `__init__` method performs some checks to ensure that the input image dimensions 
are divisible by the patch size and that the pool parameter is set to a valid value. If these checks fail, an error 
message is printed.

Next, the `__init__` method defines several layers that will be used in the forward pass of the ViT. The 
`to_patch_embedding` layer is a sequential layer that first rearranges the input tensor to group the patches together and 
then applies a linear layer to transform the patches into patch embeddings. The `pos_embedding` layer is a parameter 
tensor that is used to add positional information to the patch embeddings. The `cls_token` layer is a parameter tensor 
that represents the "class" token that will be concatenated with the patch embeddings. The `dropout` layer is a dropout 
layer that will be applied to the patch embeddings after they are concatenated with the "class" token.

The `transformer` layer is a transformer encoder that will be applied to the concatenated patch embeddings and "class" 
token. The transformer encoder applies self-attention mechanisms to the input using the specified number of layers, 
attention heads, and MLP dimensions.

The `pool` variable is used to store the value of the `pool` parameter, which determines the pooling method that will be 
used on the output of the transformer encoder. The `to_latent` layer is an identity layer that will be applied to the 
output of the transformer encoder before it is passed to the final linear layer.

Finally, the `mlp_head` layer is a sequential layer that consists of a layer normalization layer followed by a linear 
layer that maps the output of the transformer encoder to the predicted class probabilities.

Once the layers have been defined, the __init__ method is complete and the ViT is ready to process input images.
""")

# Section for Training

st.header('Training', anchor='Training')

st.subheader('Initialize Vision Transformer Model', anchor='InitializeModel')

st.write('''

''')

tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''
model = ViT(
    image_size = CFG.image_size,
    patch_size = 16,
    num_classes = CFG.num_classes,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
).to(CFG.device)
    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

st.write("""
The code you provided creates a new instance of the ViT class using the specified hyperparameters. The image_size 
parameter is set to the CFG.image_size variable, which is assumed to be defined elsewhere in the code. The patch_size 
parameter is set to 16, which means that the input images will be split into patches of size 16x16 pixels. The 
num_classes parameter is set to CFG.num_classes, which is again assumed to be defined elsewhere.

The dim parameter is set to 1024, which determines the dimension of the hidden layers in the ViT. The depth parameter 
is set to 6, which determines the number of layers in the transformer encoder. The heads parameter is set to 16, which 
determines the number of attention heads that will be used in the transformer encoder. The mlp_dim parameter is set to 
2048, which determines the dimension of the MLP layers used in the transformer encoder.

The dropout parameter is set to 0.1, which determines the dropout rate that will be used in the transformer encoder. 
The emb_dropout parameter is set to 0.1, which determines the dropout rate that will be applied to the patch embeddings 
after they are concatenated with the "class" token.

After the ViT is created, the to method is called on the instance, passing in the CFG.device variable as an argument. 
This is assumed to be a PyTorch device, such as a CPU or a GPU, which determines where the ViT will be run. This allows 
the ViT to be run on different hardware, depending on the availability and capabilities of the device.
""")
st.subheader('Image Augmentation', anchor='ImageAugmentation')

st.write('''

''')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''

    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
train_transforms = T.Compose([
        T.Resize((CFG.image_size, CFG.image_size)),
        T.AutoAugment(policy = CFG.policy),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

validation_transforms = T.Compose([
        T.Resize(CFG.image_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

test_transforms = T.Compose([
        T.Resize(CFG.image_size),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    '''
    st.code(pytorch, language)

# Section for Dataset

st.subheader('Dataset', anchor='Dataset')

st.write('''
The CIFAR-10 (Canadian Institute For Advanced Research) dataset is one of the most widely used datasets for benchmarking research in computer vision and machine learning. It is a subset of the 80 million tiny images dataset. The dataset consists of 60000 (32x32) labeled color images in 10 different classes. There are 6000 images per class. The classes are airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. The data is split into 50000 training images (five batches of 10000) and 10000 test images. The training batches contain 5000 randomnly selected images from each class while the test set includes 1000 from each.

Load the CIFAR-10 dataset from TorchVision with the following parameters: 
- `root`: a path to where the dataset is stored. We define a directory, `'./cifar_data/'`, that will be created in this case.
- `train`: specifies whether the dataset is for training or not. The train parameter should only be set to `True` for `train_dataset`. It should be set to `False` in `test_dataset`.
- `download`: whether or not to download the dataset from the internet, if it is not already available in root. If you do not already have the dataset you will want this set to `True`.
- `transform`: apply image augmentations and transformations to the dataset. Previously we defined transformations using `AutoAugment` for CIFAR-10.
''')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_1:
    pytorch = '''
train_dataset = CIFAR10(
    root = './cifar_data/',
    train = True,
    download = True,
    transform = train_transform,
)

test_dataset = CIFAR10(
    root = './cifar_data/',
    train = False,
    download = True,
    transform = test_transform,
)
    '''
    st.code(pytorch, language)

st.write('''
In order to create a validation split, we will use PyTorch's `torch.utils.data.random_split` to randomly split the CIFAR10 `test_dataset` into a new non-overlapping `validation_dataset` and `test_dataset`. The `validation_dataset` will be 80% of the data in the original test set. The new `test_dataset` will encompass the remaining 20%. Additionally, we will set a generator with the seed from the configuration to reproduce the same results.

PyTorch's `torch.utils.data.random_split` takes the parameters:
- `dataset`: The dataset which will be split. In our case we will want to split the previously defined `test_dataset`.
- `length`: The lengths for the dataset split. Here we will use an 80:20 split.
- `generator`: Used to reproduce the same split results when set with a manual seed. Use the `seed` variable defined in the configuration.
''')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_1:
    pytorch = '''
validation_dataset_size = int(len(test_dataset) * 0.8)
test_dataset_size = len(test_dataset) - validation_dataset_size
validation_dataset, test_dataset = torch.utils.data.random_split(
    test_dataset, 
    [validation_dataset_size, test_dataset_size],
    generator=torch.Generator().manual_seed(CFG.seed)    
    )
    '''
    st.code(pytorch, language)


with tab_2:
    haiku = '''

    '''
    st.code(haiku, language)

# Section for Dataloader

st.subheader('Dataloader', anchor='Dataloader')

st.write('''
Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.

The DataLoader supports both map-style and iterable-style datasets with single- or multi-process loading, customizing loading order and optional automatic batching (collation) and memory pinning.

See torch.utils.data documentation page for more details.

dataset (Dataset) – dataset from which to load the data.

batch_size (int, optional) – how many samples per batch to load (default: 1).

shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).

PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
''')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_1:
    pytorch = '''
train_loader = Dataloader(
    train_dataset, 
    batch_size = CFG.batch_size,
    shuffle = True, 
)

validation_loader = Dataloader(
    validation_dataset,
    batch_size = CFG.batch_size,
    shuffle = True,
)

test_loader = Dataloader(
    test_dataset, 
    batch_size = CFG.batch_size,
    shuffle = True, 
)
    '''
    st.code(pytorch, language)

with tab_2:
    haiku = '''

    '''
    st.code(haiku, language)

# section for loss function

st.subheader('Loss function')

st.write('''

''')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
criterion = optax.softmax_cross_entropy()
    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
criterion = nn.CrossEntropyLoss()
    '''
    st.code(pytorch, language)

# section for optimizer

st.subheader('Optimizer')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''
optimizer = optax.adam(learning_rate=CFG.learning_rate, b1=0.9, b2=0.99)
    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
    '''
    st.code(pytorch, language)

# section for learning rate scheduler

st.subheader('Learning Rate Scheduler')

tab_1, tab_2 = st.tabs(["PyTorch", "Haiku"])

with tab_2:
    haiku = '''

    '''
    st.code(haiku, language)

with tab_1:
    pytorch = '''
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    '''
    st.code(pytorch, language)

# Section for training step

st.subheader('Train Step', anchor='TrainStep')

st.write('''

''')

tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''

    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

# Section for Validation Step

st.subheader('Validation Step', anchor='ValidationStep')

st.write('''

''')

tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''

    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

# Section for Training Loop

st.subheader('Training Loop', anchor='TrainingLoop')

st.write('''

''')

tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''

    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

# Section for Saving Model

st.subheader('Save Trained Model', anchor='SaveModel')

tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''

    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

# Section for Loading Model

st.subheader('Load Trained Model', anchor='LoadModel')

tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''

    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

# Section for Making Predictions

st.subheader('Make Predictions', anchor='MakePredictions')

tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''

    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

# Section for distributed training

st.header('Distributed Training', anchor='DistributedTraining')

st.subheader('')

# Section for References

st.header('References', anchor='References')

st.write('https://www.cs.toronto.edu/~kriz/cifar.html')

# Section for Citations

st.header('Citations', anchor='Citations')




tab_pytorch, tab_haiku = st.tabs(["PyTorch", "Haiku"])

with tab_pytorch:
    pytorch = '''

    '''
    st.code(pytorch, language)

with tab_haiku:
    haiku = '''

    '''
    st.code(haiku, language)

