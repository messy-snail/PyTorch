import torch

x = torch.tensor([2.3, 4])
x = x.new_ones(5,3,dtype=torch.double) #텐서 재사용

print(x)

x = torch.randn_like(x, dtype=torch.float) #overide, 텐서를 넘겨 받음. 사이즈는 그대로 받고 랜덤하게 채움.
print(x)

#y.add_(x), 언더바 붙이면 in-place 자기 자신

print(x[:, 1])
