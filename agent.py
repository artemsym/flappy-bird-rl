import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from flappy_core import RustReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=0.001, gamma=0.99, episodes=1000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.min_lr = 0.0001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()
        self.memory = RustReplayBuffer(50000, state_dim)
        self.death_memory = RustReplayBuffer(2000, state_dim)

        self.epsilon = 1.0
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.985
        self.batch_size = 64

        self.best_score = 0
        self.stagnation = 0

    def check_plateau(self, score, patience=50, min_hr=3):
        if score > self.best_score:
            self.best_score = score
            self.stagnation = 0
        elif self.best_score >= min_hr:
            self.stagnation += 1
            if self.stagnation >= patience:
                self.lr = max(self.min_lr, round(self.lr * 0.8, 6))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
                self.stagnation = 0

    def critic_drift(self):
        with torch.no_grad():
            q_sum = sum(p.sum().item() for p in self.q_net.parameters())
            t_sum = sum(p.sum().item() for p in self.target_net.parameters())
            if abs(t_sum) < 1e-7:
                return 0.0
            return (q_sum - t_sum) / abs(t_sum) * 100.0

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        if done:
            self.death_memory.push(state, action, reward, next_state, done)

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def train_step(self, high_score=0):
        if high_score >= 10 and len(self.death_memory) >= 6 and len(self.memory) >= 58:
            s_b, a_b, r_b, ns_b, d_b = self.memory.sample_bytes(58, self.death_memory, 6)
        else:
            if len(self.memory) < self.batch_size:
                return None
            s_b, a_b, r_b, ns_b, d_b = self.memory.sample_bytes(self.batch_size)

        states = torch.frombuffer(s_b, dtype=torch.float32).view(self.batch_size, self.state_dim).to(self.device, non_blocking=True)
        actions = torch.frombuffer(a_b, dtype=torch.int64).unsqueeze(1).to(self.device, non_blocking=True)
        rewards = torch.frombuffer(r_b, dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)
        next_states = torch.frombuffer(ns_b, dtype=torch.float32).view(self.batch_size, self.state_dim).to(self.device, non_blocking=True)
        dones = torch.frombuffer(d_b, dtype=torch.float32).unsqueeze(1).to(self.device, non_blocking=True)

        current_q = self.q_net(states).gather(1, actions)

        with torch.no_grad():
            best_actions = self.q_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        tau = 0.005
        for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def set_eval_mode(self):
        self.epsilon = 0.0
        self.q_net.eval()

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.q_net.state_dict())
