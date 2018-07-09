import torch
import torch.nn as nn
import torch.nn.functional as func

#nn.Module을 상속받아 클래스를 생성
class Net(nn.Module):
    #생성자
    def __init__(self):
        #상위 클래스 생성자
        super(Net, self).__init__()
        #Conv2d(input_channel, output_channel, kernel_size, stride=1, padding=0)
        #사이즈 입력 : 하나의 int로 주어지면, height=width. tuple로 구성된 두개의 int로 주어지면, (h,w).
        #1개입력->6개출력 (5x5 conv)
        self.conv1 = nn.Conv2d(1, 6, 5)
        #6개입력->16개출력 (5x5 conv)
        self.conv2 = nn.Conv2d(6, 16, 5)

        #Fully Connected layer

        #conv2 출력인 16
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        #conv1의 결과는 activation function(ReLU)를 거치고, 2x2 max pooling 수행.
        x = func.max_pool2d(func.relu(self.conv1(x)), (2,2))
        #h=w이면 숫자 하나로 표현. 앞서 Conv2d 사이즈 입력 규칙과 동일.
        x = func.max_pool2d(func.relu(self.conv2d(x)), 2)

        #FC 들어가기 전 feature를 펴줌. -1은 나머지 차원에 의해 결정됨.
        x = x.view(-1, self.num_flat_features(x))
        x = func.relu(self.fc1(x))
        x = func.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]
        print(size)
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

#net.parameters() : hyper parameter
params = list(net.parameters())
print(len(params))
print(params[0].size())