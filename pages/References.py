import streamlit as st

st.header('Linear')

st.write("""
`nn.Linear()` is a class in PyTorch. This class represents a linear transformation of the input data. In other words, it 
applies a linear function to the input data, which can be used to map the input data to a different space. This is often 
used in the first few layers of a neural network, where it helps to extract features from the input data and compress it 
into a more compact representation that is better suited for subsequent processing by the rest of the network.
""")

st.write("""
In this equation, **x** is a vector of input data, **A** is a matrix of weights, **b** is a vector of biases, and **y** is the output of 
the linear layer. The equation says that the output of the linear layer is obtained by first multiplying the input 
vector **x** by the weight matrix **A**, which applies the linear transformation to the input data. The result is then added to 
the bias vector **b**, which shifts the output of the linear layer. The transpose of the weight matrix **A** is used in the 
equation because the dimensions of **x** and **A** must be compatible for the multiplication to be performed. The transpose of a 
matrix simply flips the matrix over its diagonal, so the rows and columns are switched, which allows the multiplication 
to be performed.
""")

st.header('LayerNorm')

st.write("""
`nn.LayerNorm` is a class in PyTorch, which is a popular deep learning framework. This class represents a layer 
normalization operation, which is a type of normalization that is applied to the inputs of a layer in a neural network. 
Normalization is a common technique used in deep learning to improve the performance and stability of a neural network. 
It helps to standardize the inputs to a layer, which can speed up training and improve the generalization of the model. 
The `nn.LayerNorm` class normalizes the input data across the specified dimensions, which can help to reduce the variance 
of the data and prevent the network from overfitting. It can be used as part of a larger model, such as a deep neural 
network, to improve its performance.
""")

st.write("""
This equation describes the layer normalization operation, where **x** is the input data, **E[x]** is the mean of the input 
data, **Var[x]** is the variance of the input data, epsilon is a small constant added to the variance to avoid division by 
zero, **gamma** and **beta** are learnable parameters of the normalization operation, and **y** is the output of the normalization.

The layer normalization operation first subtracts the mean of the input data from each element of the input, which 
centers the data around zero. It then divides the centered data by the square root of the variance of the input data, 
which scales the data so that it has unit variance. This helps to standardize the input data and make it more consistent, 
which can improve the performance of the neural network.

The **gamma** and **beta** parameters are learnable, which means that they can be adjusted during training to further improve 
the performance of the normalization operation. The **gamma** parameter is used to scale the normalized data, and the **beta** 
parameter is used to shift the normalized data. This allows the normalization operation to be adjusted to better suit 
the specific needs of the network.

The **epsilon** constant is added to the variance to avoid division by zero. It is a very small value, such as 1e-5, which 
has a negligible effect on the normalization operation but prevents numerical instability.
""")