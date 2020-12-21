import torch.nn as nn
import torch.nn.functional as F
from config import *


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()


        self.conv1 = nn.Conv2d(3, 32, 6)
        self.conv2 = nn.Conv2d(32, 64, 3)

        self.fc1 = nn.Linear(64 * 5* 5, 64)

        self.actor1 = nn.Linear(64, 64)
        self.actor2 = nn.Linear(64, 4)

        self.critic1 = nn.Linear(64, 64)
        self.critic2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        '''신경망 순전파 계산을 정의'''
        x = torch.tensor(x).float()

        x = x.to(device)

        x = F.tanh(self.conv1(x))
        x = F.tanh(self.conv2(x))

        x = x.view(-1, 64 * 5 * 5)

        x = self.dropout(torch.tanh(self.fc1(x)))

        actor_output = self.actor2(torch.tanh(self.actor1(x)))
        critic_output = self.critic2(torch.tanh(self.critic1(x)))
        #actor_output = actor_output.to('cpu') # Why???
        #critic_output = critic_output.to('cpu')

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
        # actor_output=torch.clamp_min_(actor_output,min=0)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        return torch.argmax(actor_output, dim=1)

    def get_value(self, x):
        '''상태 x로부터 상태가치를 계산'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대해 확률을 계산
        #print(actor_output)
        #print(torch.clamp(F.softmax(actor_output,dim=1),min=0.001,max=3))
        #print(actions)
        action_log_probs = log_probs.gather(1, actions.detach())  # 실제 행동의 로그 확률(log_probs)을 구함

        probs = F.softmax(actor_output, dim=1)  # dim=1이므로 행동의 종류에 대한 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy