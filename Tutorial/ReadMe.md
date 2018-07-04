PyTorch Tutorial
================
* �� Ʃ�丮���� ���Ļ���Ʈ�� �����Ͽ� �ۼ��Ͽ����ϴ�. (https://pytorch.org/tutorials/index.html)
---
#### [Install]

1. Anaconda Install
* ���̽� 3.6�� 2.7 �� ���ϴ� ������ �ٿ�ε��մϴ�. (https://www.anaconda.com/download/)
* �ش� �������� Python 3.6 vesrion(64bit) ������� �ۼ��Ͽ����ϴ�.
![Anaconda](anaconda.png)

2. PyTorch Install

* �Ƴ��ܴ� ������Ʈ�� �����ϰ�, �Ʒ��� �ڵ带 �Է��ϼ���.

      conda create -n PyTorch python=3.6  
      activate PyTorch  
      conda install pytorch cuda90 -c pytorch  
      pip install torchvision  

  * **conda create**�� ȯ�� �����ϴ� ��ɾ��Դϴ�. PyTorch �Ӹ� �ƴ϶� Tensorflow ���� �ٸ� ������ �����ӿ�ũ�� ����Ѵٰų� ���� ���̽� ������ ����ؾ��ϴ� ��� ȯ�渶�� �������ָ�, ������ð� ������ �ʾ� ���մϴ�.
  * **-n ȯ���, python=���̽����** �Է��Ͻø� �˴ϴ�. ȯ�漳�� ����Ʈ�� **conda env list**�� �Է��Ͻø� Ȯ���Ͻ� �� �ֽ��ϴ�.
  * **activate**�� �ش� ȯ���� Ȱ��ȭ ��Ű�� ��ɾ��Դϴ�. �ݴ�� ȯ���� ���������� ��ɾ�� **deactivate**�Դϴ�.
  * ���������� PyTorch�� ��ġ�ϴ� ��ɾ�� **conda install pytorch cuda90 -c pytorch**�Դϴ�. ���⼭�� **CUDA 9.0 ����� GPU ����**�� ��ġ�Ͽ����ϴ�. 
  * ��Ÿ ȯ�� ��ġ ����� �ش� ��ũ���� Ȯ���� �� �ֽ��ϴ�. (https://pytorch.org/) 
  * PyTorch�� ��� numpy�� ���������� GPU ����� ������ �����ϴ� �ټ� ������ �����ϹǷ� GPU ������ ��ġ�ϴ� ���� �����մϴ�.
  * **torchvision**�� ������ �н��� ���� ���Ǵ� �����ͼ�, ��Ʈ��ũ ����, �̹��� ��ȯ�� ���� ����� �����ϹǷ� ��ġ�ϴ� ���� �����մϴ�.

* ���������� ��ġ�ƴ��� Ȯ���ϱ� ���� �Ƴ��ܴ� ������Ʈ�� �Ʒ��� ��ɾ �Է��մϴ�.

      python  
      import torch  
      print(torch.tensor(2,2))  

![Torch Test](../../../../source/repos/PyTorch/Tutorial/torch_test.png)

* ~~(tc)�� ȯ�漳�� �̸����� ���� ������ �����ϰ� �����ߴٸ� (PyTorch)��� ����� �Ǿ��մϴ�. ���� (base)��� ��µȴٸ�, �����Ͻ� ȯ�� ������ Ȱ��ȭ �����ֽø� �˴ϴ�.~~
---
#### [Lab01.py]