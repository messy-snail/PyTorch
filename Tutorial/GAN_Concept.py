#import torch
#import torch.nn as nn

## Sequential Neural Net 생성
#D = nn.Sequential(
#    nn.Linear(784, 128), #input size, hidden size
#    nn.ReLU(),
#    nn.Linear(128,1),
#    nn.Sigmoid()
#    )

## Sequential Neural Net 생성
#G = nn.Sequential(
#    nn.Linear(100, 128),
#    nn.ReLU(),
#    nn.Linear(128, 784),
#    nn.Tanh()
#    )

#criterion = nn.BCELoss() #Binary Cross Entropy


#d_optimizer = torch.optim.Adam(D.parameters(), lr=0.01) #need beta description
#g_optimizer = torch.optim.Adam(G.parameters(), lr=0.01)

##assume x : image, z : noise distribution
#while True:
#    #train the D network
#    loss = criterion(D(x), 1) + criterion(D(G(z)), 0)
#    loss.backward() #compute gradient
#    d_optimizer.step() #find optimal parameter

#    loss = criterion(D(G(z)), 1)
#    loss.backward()
#    g_optimizer.step()
