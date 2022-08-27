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
    - [Pre-Normalization Layer](#prenorm)
    - [Feed-Forward Network](#feedforward)
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

st.subheader('Pre-Normalization Layer', anchor='prenorm')

st.write('''
Layer normalisation explicitly controls the mean and variance of individual neural network activations

Next, the output reinforced by residual connections goes through a layer normalization layer. Layer normalization, similar to batch normalization is a way to reduce the “covariate shift” in neural networks allowing them to be trained faster and achieve better performance. Covariate shift refers to changes in the distribution of neural network activations (caused by changes in the data distribution), that transpires as the model goes through model training. Such changes in the distribution hurts consistency during model training and negatively impact the model. It was introduced in the paper, “Layer Normalization” by Ba et. al. (https://arxiv.org/pdf/1607.06450.pdf).

However, layer normalization computes mean and variance (i.e. the normalization terms) of the activations in such a way that, the normalization terms are the same for every hidden unit. In other words, layer normalization has a single mean and a variance value for all the hidden units in a layer. This is in contrast to batch normalization that maintains individual mean and variance values for each hidden unit in a layer.  Moreover, unlike batch normalization, layer normalization does not average over the samples in the batch, rather leave the averaging out and have different normalization terms for different inputs. By having a mean and variance per-sample, layer normalization gets rid of the dependency on the mini-batch size. For more details about this method, please refer the original paper.

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

st.subheader('Feed-Forward Network', anchor='feedforward')



st.write("""
We propose the Gaussian Error Linear Unit (GELU), a high-performing neural
network activation function. The GELU activation function is xΦ(x), where Φ(x)
the standard Gaussian cumulative distribution function. The GELU nonlinearity
weights inputs by their value
""")
st.write("The Gaussian Error Linear Unit (GELU) is defined as:")
st.latex(r"""
\text{GELU}(x) = xP(X \leq x) = x\Phi(x) = x * {\frac 1 2} \big[1 = \text{erf}\big(x/\sqrt2\big)\big]
""")
st.write("GELU is approximated with if greater feedforward speed is worth the cost of exactness.:")
st.latex(r'''
\text{GELU}(x) = 0.5 * x * \bigg(1 + \tanh\bigg(\sqrt{\frac 2 \pi} * (x + 0.044715 * x^3)\bigg)\bigg)
''')

st.write('dropout')
st.write("""
During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. Each channel will be zeroed out independently on every forward call.

This has proven to be an effective technique for regularization and preventing the co-adaptation of neurons as described in the paper Improving neural networks by preventing co-adaptation of feature detectors .

Furthermore, the outputs are scaled by a factor of \frac{1}{1-p} 
  during training. This means that during evaluation the module simply computes an identity function.
p – probability of an element to be zeroed. Default: 0.5
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
class FeedForward(nn.Module):
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

st.subheader('Attention Mechanism', anchor='attention')

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

st.subheader('Transformer Encoder', anchor='transformer')

#st.image('')

st.write("""
The Transformer encoder (Vaswani et al., 2017) consists of alternating layers of multiheaded selfattention (MSA, see Appendix A) and MLP blocks (Eq. 2, 3). Layernorm (LN) is applied before
every block, and residual connections after every block (Wang et al., 2019; Baevski & Auli, 2019).
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
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x
    '''
    st.code(pytorch, language)

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

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

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

