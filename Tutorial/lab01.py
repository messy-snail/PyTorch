import torch

#torch.empty(r,c), torch.rand(r,c), torch.zeros(r,c,dtype)
#r : row, c: column

a = torch.tensor([2.3, 4]) #1차원 텐서 생성
b = torch.tensor([[2.2, 4.5], [5.3, 1.5]]) #2차원 텐서 생성

print("a : ", a)
print("size of a : ", a.size(), "\n")

print("b : ", b)
print("size of b : ", b.size(), "\n")

x = a.new_ones(5,3,dtype=torch.double) #텐서 재사용
print("x : ", x)

#텐서를 넘겨 받음. dtype override
x = torch.randn_like(x, dtype=torch.float) 
print("print : ", x)

#y.add_(x), 언더바 붙이면 in-place 자기 자신
c = torch.rand(2,2)
c.add_(b)
print(c)

#view와 reshape 차이점 확인 필요
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1, 8) #-1은 다른 차원에서 결정됨. 여기서는 8열이므로 2행을 가지게 됨.
print(x.size(), y.size(), z.size())
print("x: ", x, "\n")
print("y: ", y, "\n")
print("z: ", z, "\n")

#자세한 사항은 https://pytorch.org/docs/stable/torch.html 참조

#Torch tensor to NumPy
a = torch.ones(5)
print("torch tensor: ", a)
b = a.numpy()
print("numpy: ", b)

#a에 1을 더하면 numpy로 할당된 b도 같이 1이 더해짐.
a.add_(1)
print("====== Value is shared======")
print("torch tensor: ", a)
print("numpy: ", b)

#NumPy to Torch tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print("np: ", a)
print("torch: ", b)

#CUDA Tensor
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device) #tensor 할당시 device = "cuda"로 할당함.
    x = x.to(device) #cpu to gpu
    z = x+y
    print(z)
    print(z.to("cpu", torch.double)) #gpu to cpu
