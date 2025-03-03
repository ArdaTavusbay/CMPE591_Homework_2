import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import argparse
import time

from homework2 import Hw2Env

# Hyperparameters
NUM_EPISODES = 10000
UPDATE_FREQ = 4
INIT_EPS = 1.0
EPS_DECAY = 0.995
MIN_EPS = 0.05
BATCH_SIZE = 64
GAMMA = 0.99
BUFFER_SIZE = 10000
N_ACTIONS = 8

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MLPModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_dim, num_actions, lr=0.001):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = GAMMA
        self.eps = INIT_EPS
        self.min_eps = MIN_EPS
        self.eps_decay = EPS_DECAY
        self.decay_iter = 10

        self.online_net = MLPModel(state_dim, num_actions).to(device)
        self.target_net = MLPModel(state_dim, num_actions).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.memory = deque(maxlen=BUFFER_SIZE)

        self.steps_done = 0
        self.target_update_freq = 1000

    def select_action(self, state):
        if random.random() < self.eps:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_vals = self.online_net(state.unsqueeze(0))
        return int(torch.argmax(q_vals).item())

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def sample_memory(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        return states, actions, rewards, next_states

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        states, actions, rewards, next_states = self.sample_memory(batch_size)
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            best_next_actions = self.online_net(next_states).argmax(dim=1)
            target_q = self.target_net(next_states).gather(1, best_next_actions.unsqueeze(1)).squeeze(1)
            expected_q = rewards + self.gamma * target_q

        loss = nn.MSELoss()(current_q, expected_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(torch.load(path, map_location=device))
    
    def set_test_mode(self):
        self.eps = self.min_eps


def train():
    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    state_dim = len(env.high_level_state())
    agent = DQNAgent(state_dim, N_ACTIONS)

    all_rewards = []
    all_rps = []
    update_count = 0

    for ep in range(NUM_EPISODES):
        print(f"[TRAIN] Episode {ep+1}/{NUM_EPISODES}, Epsilon={agent.eps:.3f}, Steps={agent.steps_done}, Buffer size={len(agent.memory)}")
        env.reset()
        done = False
        episode_reward = 0.0
        episode_steps = 0

        state = env.high_level_state()

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            action = agent.select_action(state_tensor)
            new_state, reward, term, trunc = env.step(action)
            agent.remember(state, action, reward, new_state)

            episode_reward += reward
            episode_steps += 1
            agent.steps_done += 1

            if agent.steps_done % UPDATE_FREQ == 0:
                agent.update(BATCH_SIZE)
                update_count += 1
                if update_count % agent.decay_iter == 0:
                    agent.eps = max(agent.min_eps, agent.eps * agent.eps_decay)
                if update_count % agent.target_update_freq == 0:
                    agent.sync_target()

            state = new_state
            if term or trunc:
                done = True

        rps = episode_reward / episode_steps if episode_steps > 0 else 0.0
        all_rewards.append(episode_reward)
        all_rps.append(rps)
        print(f"[TRAIN] Episode {ep+1} ended. Reward={episode_reward:.3f}, RPS={rps:.3f}, Epsilon={agent.eps:.3f}")

    rewards_arr = np.array(all_rewards)
    np.save("reward_history.npy", rewards_arr)

    plt.figure(figsize=(10, 5))
    plt.plot(all_rps, label="Reward per Step")
    plt.xlabel("Episode")
    plt.ylabel("Reward per Step")
    plt.title("Reward per Step Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("rps_plot.png")
    plt.close()

    cumulative_reward = np.cumsum(rewards_arr)
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_reward, label="Cumulative Reward")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("cumulative_reward_plot.png")
    plt.close()

    rps_arr = np.array(all_rps)
    cumulative_rps = np.cumsum(rps_arr)
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_rps, label="Cumulative Reward per Step")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward per Step")
    plt.title("Cumulative Reward per Step Over Episodes")
    plt.legend()
    plt.grid(True)
    plt.savefig("cumulative_rps_plot.png")
    plt.close()

    agent.save("dqn_model.pth")
    print(f"[TRAIN] Training complete. Model saved to dqn_model.pth.")

    return all_rewards, all_rps


def test(model_path="dqn_model.pth", num_episodes=10, render_mode="gui"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = Hw2Env(n_actions=N_ACTIONS, render_mode=render_mode)
    state_dim = len(env.high_level_state())
    agent = DQNAgent(state_dim, N_ACTIONS)
    agent.load(model_path)
    agent.set_test_mode()
    agent.online_net.to(device)

    print(f"[TEST] Loaded model from {model_path}. Beginning test...")

    for ep in range(num_episodes):
        env.reset()
        done = False
        tot_reward = 0.0
        steps = 0
        state = env.high_level_state()

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).to(device)
            action = agent.select_action(state_tensor)
            state, reward, term, trunc = env.step(action)
            tot_reward += reward
            steps += 1
            done = term or trunc

        rps = tot_reward / steps if steps > 0 else 0.0
        print(f"Episode={ep}, Reward={tot_reward:.3f}, RPS={rps:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    args = parser.parse_args()

    if args.mode == "train":
        train()
    else:
        test()