import streamlit as st

st.set_page_config(
    page_title="Home",
)

st.title('Annotated Transformers')

st.write("A collection of annotated transformer architectures in Meta's [PyTorch](https://pytorch.org/) and DeepMind's "
         "[Haiku](https://github.com/deepmind/dm-haiku), [Optax](https://github.com/deepmind/optax), and Google's [JAX](https://jax.readthedocs.io/en/latest/index.html).")

st.header("[Vision Transformer](Vision_Transformer)")
st.markdown('''
- [Research Paper](https://arxiv.org/abs/2010.11929)
- [Official Repository](https://github.com/google-research/vision_transformer)
''')

## Citations
st.header("Citations")
st.markdown('''
@article{DBLP:journals/corr/abs-2010-11929,
  author    = {Alexey Dosovitskiy and
               Lucas Beyer and
               Alexander Kolesnikov and
               Dirk Weissenborn and
               Xiaohua Zhai and
               Thomas Unterthiner and
               Mostafa Dehghani and
               Matthias Minderer and
               Georg Heigold and
               Sylvain Gelly and
               Jakob Uszkoreit and
               Neil Houlsby},
  title     = {An Image is Worth 16x16 Words: Transformers for Image Recognition
               at Scale},
  journal   = {CoRR},
  volume    = {abs/2010.11929},
  year      = {2020},
  url       = {https://arxiv.org/abs/2010.11929},
  eprinttype = {arXiv},
  eprint    = {2010.11929},
  timestamp = {Fri, 20 Nov 2020 14:04:05 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2010-11929.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}''')