import torch.nn as nn
import torch.nn.functional as F
from config import *
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # self.inception=Inception3().cuda()
        self.conv1 = nn.Conv2d(3, 32, 5,padding=2)

        self.conv2 = nn.Conv2d(32, 64, 5)


        self.fc1 = nn.Linear(64*8*8, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(128, 64)
        self.critic2 = nn.Linear(64, 32)

        self.dropout = nn.Dropout(p=0.4)
        self.activation = self.mish

        # self.activation=torch.tanh
    def forward(self, x):
        '''신경망 순전파 계산을 정의'''

        x = x.to(device)
        #
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        # print(x.size())
        # x = self.inception(x)

        # print(x.size())
        x = x.view(-1, 64*8*8)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        x = self.dropout(self.activation(self.fc3(x)))
        x = self.dropout(self.activation(self.fc4(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))

        return critic_output, actor_output

    def act(self, x):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        # actor_output=torch.clamp_min_(actor_output,min=0)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        return action

    def deterministic_act(self, x):
        '''상태 x로부터 행동을 확률적으로 결정'''
        value, actor_output = self(x)
        return torch.argmax(actor_output, dim=1)

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x)

        return value, actor_output

    def evaluate_actions(self, x, actions):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions.detach())  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy, actor_output

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))


class Net2(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, stride=2, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3,padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3,padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.projection = nn.Conv2d(32, 64, 1, stride=2)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)

        self.actor1 = nn.Linear(64, 16)
        self.actor2 = nn.Linear(16, 4)

        self.critic1 = nn.Linear(64, 32)
        self.critic2 = nn.Linear(32, 8)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.mish

        # self.activation=torch.tanh
    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = self.conv1(x)
        x = self.pool1(x)

        idx = x

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x)+idx)

        idx = x

        x = self.activation(self.conv4(idx))
        idx = self.projection(idx)

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x) + idx)

        x = self.pool2(x)

        x = x.view(-1, 64)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))

        actor_output = self.activation(self.actor1(x))
        actor_output = self.actor2(actor_output)

        critic_output = self.activation(self.critic1(x))
        critic_output = self.critic2(critic_output)

        return critic_output, actor_output


class Net3(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3,padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3,padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3,padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool=nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.conv7=nn.Conv2d(64,64,7,padding=3, stride=2)

        self.fc1 = nn.Linear(64*3*3, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(128, 128)
        self.critic2 = nn.Linear(128, 64)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.mish

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = self.activation(self.conv1(x))

        idx = x

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x)+idx)

        x = self.activation(self.conv4(x))

        idx = x

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x)+idx)

        x = self.pool(x)

        x = self.activation(self.conv7(x))

        x = x.view(-1, 64*3*3)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))

        return critic_output, actor_output
