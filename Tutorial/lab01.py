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

print(x)

#overide, 텐서를 넘겨 받음. 사이즈는 그대로 받고 랜덤하게 채움.
x = torch.randn_like(x, dtype=torch.float) 

print(x)

#y.add_(x), 언더바 붙이면 in-place 자기 자신

print(x[:, 1])
