"""
DQN for discrete action spaces.

Key idea (1-step bootstrapping):
  y = r + gamma * (1 - done) * max_a' Q_target(s', a')

This file includes:
- MLP helper
- QNetwork
- ReplayBuffer storing (s, a, r, s2, done)
- DQNConfig
- DQNAgent (choose_action, store_transition, train, save, load)

Notes:
- For stability: target network, replay buffer, epsilon-greedy, gradient clipping optional.
- This is "vanilla DQN". You can extend to Double DQN / Dueling easily.
"""

import random
from dataclasses import dataclass
from typing import Deque, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------
# Small MLP helper
# ---------------------------
def mlp(sizes, activation=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else out_act
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)


# ---------------------------
# Q Network
# ---------------------------
class QNetwork(nn.Module):
    """
    Outputs Q-values for all discrete actions:
      Q(s) -> shape [batch, n_actions]
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128), activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions], activation=activation, out_act=nn.Identity)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------
# Replay Buffer
# ---------------------------
class ReplayBuffer:
    """
    Stores tuples:
      (s, a, r, s2, done)
    where a is an int action index.
    """
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=capacity)

    def push(self, s, a: int, r: float, s2, done: bool):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            int(a),
            float(r),
            np.asarray(s2, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),      # action indices
            np.asarray(r, dtype=np.float32),
            np.asarray(s2, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------
# Config
# ---------------------------
@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-4
    buffer_capacity: int = 100_000
    hidden: Tuple[int, int] = (128, 128)
    batch_size: int = 128

    # epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000  # linear decay over these many action selections

    # target network update
    target_update_every: int = 1000  # steps

    # misc
    grad_clip_norm: Optional[float] = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0


# ---------------------------
# DQN Agent
# ---------------------------
class DQNAgent:
    def __init__(self, name, obs_dim: int, n_actions: int, cfg: DQNConfig):
        self.name = name
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.n_actions = n_actions

        # seeds
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # networks
        self.q = QNetwork(obs_dim, n_actions, hidden=cfg.hidden).to(self.device)
        self.q_target = QNetwork(obs_dim, n_actions, hidden=cfg.hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # optimizer
        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.lr)

        # replay
        self.buffer = ReplayBuffer(cfg.buffer_capacity)

        # epsilon schedule state
        self.step_count = 0

    def epsilon(self) -> float:
        # linear decay
        if self.cfg.eps_decay_steps <= 0:
            return self.cfg.eps_end
        frac = min(1.0, self.step_count / float(self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def choose_action(self, state, greedy: bool = False) -> int:
        """
        Epsilon-greedy action selection.
        If greedy=True, always argmax.
        """
        self.step_count += 1
        eps = 0.0 if greedy else self.epsilon()

        if random.random() < eps:
            return random.randrange(self.n_actions)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q(s)  # [1, n_actions]
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action: int, reward: float, next_state, done: bool):
        self.buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)  # [B,1]
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)  # [B,1]

        # Q(s,a)
        q_sa = self.q(states).gather(1, actions)  # [B,1]

        # target: r + gamma*(1-done)*max_a' Q_target(s',a')
        with torch.no_grad():
            q_next_max = self.q_target(next_states).max(dim=1, keepdim=True)[0]  # [B,1]
            y = rewards + self.gamma * (1.0 - dones) * q_next_max

        loss = nn.MSELoss()(q_sa, y)

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        # hard update target periodically
        if (self.step_count % self.cfg.target_update_every) == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return {"loss": float(loss.item()), "eps": float(self.epsilon())}

    def save(self, path: str):
        try:
            torch.save({
                "q": self.q.state_dict(),
                "q_target": self.q_target.state_dict(),
                "opt": self.optimizer.state_dict(),
                "cfg": self.cfg.__dict__,
                "step_count": self.step_count,
                "mode": "dqn",
            }, path)
            return f"Model saved successfully at: {path}"
        except Exception as e:
            return f"Error saving model to {path}:: Error: {e}"

    def load(self, path: str, map_location: Optional[str] = None):
        try:
            if map_location is None:
                map_location = self.device

            ckpt = torch.load(path, map_location=map_location, weights_only=False)
            self.q.load_state_dict(ckpt["q"])
            self.q_target.load_state_dict(ckpt["q_target"])
            self.optimizer.load_state_dict(ckpt["opt"])
            self.step_count = int(ckpt.get("step_count", 0))

            # ensure optimizer tensors are on correct device
            for st in self.optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(self.device)

            return f"Model loaded successfully from: {path}"
        except Exception as e:
            return f"Error loading model from {path}: {e}"