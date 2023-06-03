import os
from typing import Callable, NewType

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

ActivationFunction = NewType('ActivationFunction', Callable[[np.ndarray], np.ndarray])

sigmoid = ActivationFunction(lambda X: 1.0 / (1.0 + np.exp(-X)))
tanh = ActivationFunction(lambda X: np.tanh(X))
relu = ActivationFunction(lambda X: np.maximum(0, X))
leaky_relu = ActivationFunction(lambda X: np.where(X > 0, X, X * 0.01))
linear = ActivationFunction(lambda X: X)
softmax = ActivationFunction(lambda X: np.exp(X) / np.sum(np.exp(X), axis=0))


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(LinearDeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.file_name = name

        # print(input_dims)
        # r4
        print("NN", *input_dims)
        self.fc1 = nn.Linear(*input_dims, 32)
        self.fc2 = nn.Linear(32, 64)
        self.q = nn.Linear(64, n_actions)
        # r8 32
        # self.fc1 = nn.Linear(*input_dims, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.q = nn.Linear(64, n_actions)

        # self.fc1 = nn.Linear(*input_dims, 128)
        # self.fc2 = nn.Linear(128, 128)
        # self.q = nn.Linear(128, n_actions)
        # r64
        # self.fc1 = nn.Linear(*input_dims, 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 64)
        # self.q = nn.Linear(64, n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        # flat3 = F.relu(self.fc2(flat2))
        q = self.q(flat2)

        return q

    def save_checkpoint(self, generation: int, score: int):
        print(f"... save ... at {self.checkpoint_file}")
        folder = os.path.join(self.checkpoint_dir, f"snake_{generation}_{score}")
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.checkpoint_file = os.path.join(folder, self.file_name)
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print(f"... load ... from {self.checkpoint_file}")
        self.load_state_dict(T.load(self.checkpoint_file))


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # print(input_dims[0])
        self.conv1 = nn.Conv3d(input_dims[0], 32, 1, stride=1)
        # self.conv2 = nn.Conv3d(32, 64, 3, stride=1)
        # self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        # conv2 = F.relu(self.conv2(conv1))
        # conv3 = F.relu(self.conv3(conv2))
        conv_state = conv1.view(conv1.size()[0], -1)

        flat1 = F.relu(self.fc1(conv_state))
        q = self.fc2(flat1)

        return q

    def calculate_conv_output_dims(self, input_dims):
        # print(input_dims)
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        # dims = self.conv2(dims)
        # dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def save_checkpoint(self):
        print("... save ...")

        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print("... load ...")
        self.load_state_dict(T.load(self.checkpoint_file))


if __name__ == '__main__':
    q_eval = LinearDeepQNetwork(0.05, 4,
                                input_dims=[32, ],
                                name="some",
                                chkpt_dir="")
    # for l in range(1, L):
    #     W = self.params['W' + str(l)]
    #     b = self.params['b' + str(l)]
    #     Z = np.dot(W, A_prev) + b
    #     # print("hidden")
    #     A_prev = self.hidden_activation(Z)
    #     self.params['A' + str(l)] = A_prev
    A_prev = np.zeros(shape=(32, 1))
    weights = list(q_eval.parameters())
    for i in range(0, len(weights), 2):
        W = weights[i].cpu().detach().numpy()
        b = np.array([[w] for w in weights[i + 1].cpu().detach().numpy()])
        # print(W.shape)
        # print(b.shape)
        # print(A_prev.shape)
        z = np.dot(W, A_prev) + b
        if i == len(weights) - 2:

            A_prev = softmax(z)
        else:
            A_prev = relu(z)
        print(A_prev)
        # print(t.cpu().detach().numpy())
    # print(q_eval.cpu().numpy().array([0]))
