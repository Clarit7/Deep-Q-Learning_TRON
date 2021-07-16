import torch.nn as nn
import torch.nn.functional as F
from config import *
import torchvision.models as models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden_dim = 32

        self.conv1 = nn.Conv2d(2, self.hidden_dim, 3, padding=1)

        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv6 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv7 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)

        self.conv_actor = nn.Conv2d(self.hidden_dim, 2, 1)
        self.fc_actor1 = nn.Linear(2 * 12 * 12, 4)

        self.conv_critic = nn.Conv2d(self.hidden_dim, 1, 1)
        self.fc_critic1 = nn.Linear(12 * 12, 128)
        self.fc_critic2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = self.relu(self.conv1(x))

        shortcut = x
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x) + shortcut)

        actor_output = self.conv_actor(x).view(-1, 2 * 12 * 12)
        actor_output = self.fc_actor1(self.relu(actor_output))

        critic_output = self.conv_critic(x).view(-1, 12 * 12)
        critic_output = self.fc_critic1(self.relu(critic_output))
        critic_output = self.fc_critic2(self.relu(critic_output))

        return critic_output, actor_output

    def get_pv(self, x):
        Q, policy = self(x)
        policy = F.softmax(policy, dim=1).clamp(max=1 - 1e-20)  # Prevent 1s and hence NaNs
        V = (Q * policy).sum(1, keepdim=True)  # V is expectation of Q under π

        return policy, Q, V

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

        return value

    def evaluate_actions(self, x, actions):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions.detach())  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy

    def mish(self, x):
        return x * torch.tanh(F.softplus(x))


class NetStatic10(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden_dim = 32

        self.conv1 = nn.Conv2d(2, self.hidden_dim, 3, padding=1)

        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv6 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv7 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)

        self.conv_actor = nn.Conv2d(self.hidden_dim, 2, 1)
        self.fc_actor1 = nn.Linear(2 * 12 * 12, 128)
        self.fc_actor2 = nn.Linear(128, 4)

        self.conv_critic = nn.Conv2d(self.hidden_dim, 1, 1)
        self.fc_critic1 = nn.Linear(12 * 12, 128)
        self.fc_critic2 = nn.Linear(128, 4)

        self.relu = nn.ReLU()

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        # x = x.to(device)
        x = self.relu(self.conv1(x))

        shortcut = x
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x) + shortcut)

        actor_output = self.conv_actor(x).view(-1, 2 * 12 * 12)
        actor_output = F.dropout(self.fc_actor1(self.relu(actor_output)), 0.1)
        actor_output = self.fc_actor2(self.relu(actor_output))

        critic_output = self.conv_critic(x).view(-1, 12 * 12)
        critic_output = F.dropout(self.fc_critic1(self.relu(critic_output)), 0.1)
        critic_output = self.fc_critic2(self.relu(critic_output))
        return critic_output, actor_output

class NetStatic15(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden_dim = 128

        self.conv1 = nn.Conv2d(2, self.hidden_dim, 3, padding=1)

        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv6 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv7 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv8 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv9 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv10 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv11 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv12 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv13 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv14 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv15 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv16 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv17 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)

        self.conv_actor = nn.Conv2d(self.hidden_dim, 2, 1)
        self.fc_actor1 = nn.Linear(2 * 17 * 17, 512)
        self.fc_actor2 = nn.Linear(512, 128)
        self.fc_actor3 = nn.Linear(128, 4)

        self.conv_critic = nn.Conv2d(self.hidden_dim, 1, 1)
        self.fc_critic1 = nn.Linear(17 * 17, 256)
        self.fc_critic2 = nn.Linear(256, 64)
        self.fc_critic3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = self.relu(self.conv1(x))

        shortcut = x
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv14(x))
        x = self.relu(self.conv15(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv16(x))
        x = self.relu(self.conv17(x) + shortcut)

        actor_output = self.conv_actor(x).view(-1, 2 * 17 * 17)
        actor_output = self.fc_actor1(self.dropout(self.relu(actor_output)))
        actor_output = self.fc_actor2(self.dropout(self.relu(actor_output)))
        actor_output = self.fc_actor3(self.dropout(self.relu(actor_output)))

        critic_output = self.conv_critic(x).view(-1, 17 * 17)
        critic_output = self.fc_critic1(self.dropout(self.relu(critic_output)))
        critic_output = self.fc_critic2(self.dropout(self.relu(critic_output)))
        critic_output = self.fc_critic3(self.dropout(self.relu(critic_output)))

        return critic_output, actor_output

class NetStatic20(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.hidden_dim = 32

        self.conv1 = nn.Conv2d(2, self.hidden_dim, 3, padding=1)

        self.conv2 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv4 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv6 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv7 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv8 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv9 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv10 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)
        self.conv11 = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, padding=1)

        self.conv_actor = nn.Conv2d(self.hidden_dim, 2, 1)
        self.fc_actor1 = nn.Linear(2 * 22 * 22, 256)
        self.fc_actor2 = nn.Linear(256, 4)

        self.conv_critic = nn.Conv2d(self.hidden_dim, 1, 1)
        self.fc_critic1 = nn.Linear(22 * 22, 256)
        self.fc_critic2 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = x.to(device)

        x = self.relu(self.conv1(x))

        shortcut = x
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x) + shortcut)
        shortcut = x
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x) + shortcut)

        actor_output = self.conv_actor(x).view(-1, 2 * 22 * 22)
        actor_output = F.dropout(self.fc_actor1(self.relu(actor_output)), 0.1)
        actor_output = self.fc_actor2(self.relu(actor_output))

        critic_output = self.conv_critic(x).view(-1, 22 * 22)
        critic_output = F.dropout(self.fc_critic1(self.relu(critic_output)), 0.1)
        critic_output = self.fc_critic2(self.relu(critic_output))

        return critic_output, actor_output

class Net6(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3,padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3,padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3,padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv7 = nn.Conv2d(64, 64, 7, padding=3, stride=2)

        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.fc1 = nn.Linear(64 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(128, 64)
        self.critic2 = nn.Linear(64, 16)
        self.critic3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.2)
        # self.activation = torch.nn.Tanh()
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

        x = self.activation(self.conv7(x))

        x = self.pool(x)

        x = x.view(-1, 64 * 2 * 2)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))

        return critic_output, actor_output



class Net8(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv7 = nn.Conv2d(64, 64, 7, padding=3, stride=2)

        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(128, 64)
        self.critic2 = nn.Linear(64, 16)
        self.critic3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.mish

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''

        x = x.to(device)

        x = self.activation(self.conv1(x))

        idx = x

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x) + idx)

        x = self.activation(self.conv4(x))

        idx = x

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x) + idx)

        x = self.activation(self.conv7(x))

        x = self.pool(x)

        x = x.view(-1, 64 * 3 * 3)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))

        return critic_output, actor_output


class Net10(Net):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)

        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=1)

        self.conv7 = nn.Conv2d(64, 64, 7, padding=3, stride=2)

        self.pool = nn.AvgPool2d(kernel_size=3, padding=1, stride=2)

        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 128)

        self.actor1 = nn.Linear(128, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(128, 64)
        self.critic2 = nn.Linear(64, 16)
        self.critic3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(p=0.2)
        self.activation = self.mish

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''

        x = x.to(device)

        x = self.activation(self.conv1(x))

        idx = x

        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x) + idx)

        x = self.activation(self.conv4(x))

        idx = x

        x = self.activation(self.conv5(x))
        x = self.activation(self.conv6(x) + idx)

        x = self.activation(self.conv7(x))

        x = self.pool(x)

        x = x.view(-1, 64 * 3 * 3)

        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))

        actor_output = self.actor2(self.activation(self.actor1(x)))

        critic_output = self.critic2(self.activation(self.critic1(x)))
        critic_output = self.critic3(self.activation(critic_output))

        return critic_output, actor_output