from torch import nn

class PendulumPolicyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_net = nn.Sequential(nn.Linear(3,64),
                                       nn.ReLU(),
                                       nn.Linear(64,1),
                                       nn.Tanh())

    def forward(self, x):
        return self.policy_net(x) * 2

class ContiniousLunarLanderNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_net = nn.Sequential(nn.Linear(8,64),
                                        nn.ReLU(),
                                        nn.Linear(64,64),
                                        nn.ReLU(),
                                        nn.Linear(64,2),
                                        nn.Tanh())

    def forward(self, x):
        return self.policy_net(x)

class BipedalWalkerNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_net = nn.Sequential(nn.Linear(24,64),
                                        nn.ReLU(),
                                        nn.Linear(64,64),
                                        nn.ReLU(),
                                        nn.Linear(64,4),
                                        nn.Tanh())

    def forward(self, x):
        return self.policy_net(x)

class HalfCheetahNet(nn.Module):
    def __init__(self):
        super(HalfCheetahNet, self).__init__()
        self.policy_net = nn.Sequential(nn.Linear(17,64),
                                       nn.ReLU(),
                                       nn.Linear(64,64),
                                       nn.ReLU(),
                                       nn.Linear(64,6),
                                       nn.Tanh())
        
    def forward(self, x):
        return self.policy_net(x)

class HalfCheetahNetLarge(nn.Module):
    def __init__(self):
        super(HalfCheetahNetLarge, self).__init__()
        self.policy_net = nn.Sequential(nn.Linear(17,128),
                                       nn.ReLU(),
                                       nn.Linear(128,128),
                                       nn.ReLU(),
                                       nn.Linear(128,6),
                                       nn.Tanh())
        
    def forward(self, x):
        return self.policy_net(x)

class RandomNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.policy_net = nn.Sequential(nn.Linear(10,64),
                                        nn.ReLU(),
                                        nn.Linear(64,64),
                                        nn.ReLU(),
                                        nn.Linear(64,5),
                                        nn.Tanh())

    def forward(self, x):
        return self.policy_net(x)

class SwimmerNet(nn.Module):
    def __init__(self):
        super(SwimmerNet, self).__init__()
        self.policy_net = nn.Sequential(nn.Linear(8,64),
                                        nn.ReLU(),
                                        nn.Linear(64,64),
                                        nn.ReLU(),
                                        nn.Linear(64,2),
                                        nn.Tanh())

    def forward(self, x):
        return self.policy_net(x)

class FetchPickAndPlaceNet(nn.Module):
    def __init__(self):
        super(FetchPickAndPlaceNet, self).__init__()
        self.policy_net = nn.Sequential(nn.Linear(28,128),
                                        nn.ReLU(),
                                        nn.Linear(128,128),
                                        nn.ReLU(),
                                        nn.Linear(128,4))
                                
    def forward(self, x):
        return self.policy_net(x)

class FetchReachNet(nn.Module):
    def __init__(self):
        super(FetchReachNet, self).__init__()
        self.policy_net = nn.Sequential(nn.Linear(13,64),
                                        nn.ReLU(),
                                        nn.Linear(64,4))
                                
    def forward(self, x):
        return self.policy_net(x)