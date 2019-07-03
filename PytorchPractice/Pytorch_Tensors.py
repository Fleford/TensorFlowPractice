# -*- coding: utf-8 -*-

import torch

torch.set_printoptions(linewidth=120)


dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)
print(x.shape)

# Randomly initialize weights
# Note that requires_grad=True was set
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(1000):
    # Forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call w1.grad and w2.grad will be Tensors holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():   # Perform the following without accumulating gradients
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # Zero out gradients
        w1.grad.zero_()
        w2.grad.zero_()
