import random
from collections import deque
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim=4, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(states, dtype=torch.float32, device=device),
            torch.as_tensor(actions, dtype=torch.long, device=device),
            torch.as_tensor(rewards, dtype=torch.float32, device=device),
            torch.as_tensor(next_states, dtype=torch.float32, device=device),
            torch.as_tensor(dones, dtype=torch.float32, device=device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim=4, action_dim=2, lr=1e-3, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Обучаемая и целевая сети
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss стабильнее MSE

        self.memory = ReplayBuffer(capacity=50000)
        self.batch_size = 64
        self.gamma = gamma

        # Исследование (decay происходит один раз за ЭПИЗОД, а не за шаг!)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state, eval_mode=False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def train_step(self):
        """Один шаг оптимизации. Возвращает float значение loss или None."""
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size, self.device)
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        # Текущая оценка Q(s, a)
        current_q = self.q_net(states).gather(1, actions)

        # Таргет Беллмана
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Ограничение нормы градиента для защиты от взрыва градиентов
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())

    def decay_epsilon(self):
        """Снижение вероятности случайного действия в конце каждого эпизода."""
        if self.epsilon > self.epsilon_min:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        """Синхронизация весов целевой сети."""
        self.target_net.load_state_dict(self.q_net.state_dict())

    def set_eval_mode(self):
        """Режим оценки/игры: отключаем случайные действия."""
        self.epsilon = 0.0
        self.q_net.eval()

    def save(self, filepath, episode=0, best_score=0):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.q_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "episode": episode,
            "best_score": best_score,
            "epsilon": self.epsilon,
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.q_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net.load_state_dict(checkpoint.get("target_state_dict", checkpoint["model_state_dict"]))
            self.epsilon = checkpoint.get("epsilon", 0.0)
            return checkpoint.get("episode", 0), checkpoint.get("best_score", 0)
        else:
            # Поддержка загрузки чистых state_dict
            self.q_net.load_state_dict(checkpoint)
            self.target_net.load_state_dict(checkpoint)
            return 0, 0
