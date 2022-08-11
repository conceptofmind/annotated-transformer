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
- [Prerequisites](#Prerequisites)
    - [Installs](#Installs)
    - [Imports](#Imports)
- [Model](#Model)
- [Training](#Training)
- [References](#References)
- [Citations](#Citations) 
''')

# Section for Introduction

st.header('Introduction', anchor='Introduction')

st.markdown('''"
While the Transformer architecture has become the de-facto standard for natural language processing tasks, its applications to computer vision remain limited. In vision, attention is either applied in conjunction with convolutional networks, or used to replace certain components of convolutional networks while keeping their overall structure in place. We show that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
" - Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby''')

# Section for Prereqs

st.header('Prerequisites', anchor='Prerequisites')

st.subheader('Installs', anchor='Installs')

st.subheader('Imports', anchor='Imports')

imports_tab_1, imports_tab_2 = st.tabs(["Haiku", "PyTorch"])

with imports_tab_1:
    haiku_imports = '''
from functools import partial

import haiku as hk
from haiku import PRNGSequence

import jax
from jax import random, nn
import jax.numpy as jnp

from einops import rearrange, repeat
    '''
    st.code(haiku_imports, language)

with imports_tab_2:
    pytorch_imports = '''
import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
    '''
    st.code(pytorch_imports, language)

# Section for Model

st.header('Model', anchor='Model')

# Section for Training

st.header('Training', anchor='Training')

# Section for References

st.header('References', anchor='References')

# Section for Citations

st.header('Citations', anchor='Citations')