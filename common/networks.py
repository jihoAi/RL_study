import torch.nn as nn


class MLP(nn.Module):
    """ReLU 은닉층을 가진 MLP. 출력층에는 활성화 함수를 쓰지 않는다.

    예) MLP(4, 2, hidden=(8, 4))  ->  4 -> 8 -> 4 -> 2
    """

    def __init__(self, in_dim, out_dim, hidden=(8, 4)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
