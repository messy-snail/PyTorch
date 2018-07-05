import torch as tc
import matplotlib.pyplot as plt

num_points = 1000
x = tc.rand(num_points)

y = x*0.4+tc.rand(num_points)

plt.plot(x.numpy(),y.numpy())
plt.show()