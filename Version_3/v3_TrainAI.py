import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from collections import namedtuple
from game_v3 import Game2048
from random import randint

# Hyperparameters
batch_size = 64
learning_rate = 1e-6
gamma = 0.99
initial_epsilon = 0.1
final_epsilon = 1e-4
num_iters = 256

eps = []
history = []

# Device selection
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print('Using {} device'.format(device))

# Deep Q-Network model
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4 * 4, 16 * 16),
            nn.ReLU(),
            nn.Linear(16 * 16, 16 * 6),
            nn.ReLU(),
            nn.Linear(16 * 6, 16),
            nn.ReLU(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

# Memory buffer for experience replay
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Function to calculate reward
def calc_reward(state, last_total):
    empties = 0
    total = 0
    highest = 0 
    for i in state:
        if i == 0:
            empties += 1
        else:
            total += i
        highest = max(highest, i)
    return total - last_total - 0.5, total

# Training function
def train(episodes):
    mem = ReplayMemory(500000)
    model = DQN().to(device)
    last_total = 0
    losses = []

    for episode in range(episodes):
        moves = ['w', 'a', 's', 'd']
        if episode % 10 == 0:
            new_game = Game2048(human=False, printboard=True)
        else:
            new_game = Game2048(human=False, printboard=False)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        next_state = torch.Tensor(np.array([[0 for _ in range(16)]]))
        output = [0 for _ in range(16)]

        while type(output) == list:
            state = next_state.squeeze().to(device)
            pred = model(state)

            legal_moves = new_game.getLegalMoves()
            for _ in range(4):
                action = torch.argmax(pred)
                if legal_moves[action] == 1:
                    break
                else:
                    pred[action] = 0

            epsilon = final_epsilon + ((episodes - episode) * (initial_epsilon - final_epsilon) / episodes)
            take_random_action = random.random() <= epsilon
            if take_random_action:
                move = moves[random.randint(0, 3)]
            else:
                move = moves[action.item()]

            output = new_game.step(move)

            if new_game.getTerminal():
                reward = -1
                output = new_game.getBoard()
            else:
                reward, last_total = calc_reward(next_state.squeeze(), last_total)
            next_state = torch.Tensor(np.array([[math.log2(i) if i != 0 else i for i in output]]))

            if new_game.getTurn() > 3000:
                mem.push(state.unsqueeze(0), action, next_state, reward)

            if len(mem) > 256:
                transitions = mem.sample(batch_size)
                batch = Transition(*zip(*transitions))

                non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                next_state_batch = torch.cat(batch.next_state)
                state_batch = torch.cat(batch.state)
                action_batch = torch.Tensor(batch.action)
                reward_batch = torch.Tensor(batch.reward)

                current_prediction_batch = model(state_batch).to(device)
                next_prediction_batch = model(next_state_batch).unsqueeze(0).to(device)
                y_batch = torch.stack(tuple(reward + gamma * torch.max(prediction) for reward, prediction in zip(reward_batch, next_prediction_batch)))

                myAction = np.array(action_batch)
                actions = []
                actions_set = {
                    0:[[1, 0, 0, 0]],
                    1:[[0, 1, 0, 0]],
                    2:[[0, 0, 1, 0]],
                    3:[[0, 0, 0, 1]]
                }
                for i in myAction:
                    actions.append(torch.Tensor(actions_set[int(i)]))
                action_batch = torch.cat(actions)

                q_value = torch.sum(current_prediction_batch * action_batch, dim=-1)
                optimizer.zero_grad()
                loss = criterion(q_value, y_batch)
                loss.backward()
                optimizer.step()

                losses.append(loss)

                if new_game.getTurn() % 100 == 0:
                    print("Episode: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
                        episode + 1,
                        num_iters,
                        action,
                        loss,
                        epsilon, reward, torch.max(pred)))
            
            if new_game.getTerminal():
                break

        if episode % 30 == 0:
            checkpoint_path = "v1_TrainAI_ep{}.pth".format(int(episode))
            torch.save({
                'checkpoint_episode': episode,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'average_length': history,
                'losses': losses
            }, checkpoint_path)
        
        eps.append(episode)
        history.append(new_game.getTurn())

    checkpoint_path = "wv3_TrainAI_ep{}.pth".format(int(episode))
    torch.save({
        'checkpoint_episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'average_length': history,
        'losses': losses
    }, checkpoint_path)

    plt.plot(np.array(eps), np.array(history))
    plt.show()

train(num_iters)