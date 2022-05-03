import torch
import numpy as np

'''
================= Tensors ==================
'''
## Initialzing a Tensor ##

# Directly from Data 
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)

# From a NumPy array -
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
print(x_np)

# From another tensor -
x_ones = torch.ones_like(x_data)
print(f"Ones Tensor: \n {x_ones} \n")
x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# With random or constant values -
shape = (2, 3, )
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

## Attributes of a Tensor ##
tensor = torch.rand(3, 4)
print(tensor)
print('\n')
print(f"Shape of tensor : {tensor.shape}")
print(f"Datatype of tensor : {tensor.dtype}")
print(f"Device tensor is stored on : {tensor.device}")

## Operations on Tensors ##
# We move our tensor to the GPU if available -
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# some other operations -
tensor = torch.ones(4, 4)
print(tensor)
print('First Row: ', tensor[0])
print('First Column: ', tensor[:, 0])
print('Last Column: ', tensor[..., -1])
tensor[:, 1] = 0
print(tensor)

# Joining Tensors -
t1 = torch.cat([tensor, tensor, tensor], dim = 1)
print(t1)

# Arithmetic Operations -
# matrix multiplication between two tensors
y1 = tensor @ tensor.T 
y2 = tensor.matmul(tensor.T)
print(y1)
print(y2)
y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out = y3)
print(y3)

# This computes the element-wise product.
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)
print(z1)
print(z2)
print(z3)

# Single element tensors. If you have a one-element tensor, for example by aggregating
# all values of a tensor into one value, you can convert it to a Python numerical value
# using item()
print('\n')
print(tensor)
agg = tensor.sum()
print(agg, type(agg))
agg_item = agg.item()
print(agg_item, type(agg_item))

# In-place Operations (it can change the history and can impact the back-propagation)
print(tensor, "\n")
tensor.add_(5) # adding 5 to all the values of the tensor in-place.
print(tensor)

# Tensors on the CPU and NumPy arrays can share their underlying memory locations, 
# and changing one will change the other.

# Tensor to NumPy array -
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

# A change in the tensor reflects in the NumPy array -
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")

# NumPy array to Tensor
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")