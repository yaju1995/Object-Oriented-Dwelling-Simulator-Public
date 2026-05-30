"""
DDPG with Monte Carlo (episodic) returns for continuous action spaces.

Key idea (no bootstrapping):
  y = G_t
where:
  G_t = r_t + gamma r_{t+1} + ... + gamma^{T-1-t} r_{T-1}
and T is the terminal step of the episode.

This file includes:
- MLP helper
- Actor / Critic networks
- ReplayBuffer that stores (s, a, G, s_terminal, done=1, gamma_pow=0)
- EpisodeReturnAdder that converts 1-step env transitions into MC-return transitions
- DDPGConfig
- DDPGAgent (choose_action, store_transition, train, save, load)

Notes:
- Monte Carlo targets can be high-variance. You’ll often want larger batches, reward normalization,
  and/or using TD(lambda) / n-step instead for stability.
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


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64),
                 activation=(nn.ReLU, nn.Tanh)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, act_dim],
                       activation=activation[0],
                       out_act=activation[1])  # tanh => [-1,1]

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64),
                 activation=(nn.ReLU, nn.Tanh)):
        super().__init__()
        self.net = mlp([obs_dim + act_dim, *hidden, 1],
                       activation=activation[0],
                       out_act=nn.Identity)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=1)
        return self.net(x)


# ---------------------------
# Replay Buffer (MC-return)
# ---------------------------
class ReplayBuffer:
    """
    Stores tuples:
      (s, a, G, sT, doneT, gamma_pow)

    For Monte Carlo episodic targets we store:
      doneT = 1.0
      gamma_pow = 0.0
    so the generic target formula
      y = G + gamma_pow*(1-doneT)*Q_target(...)
    reduces to y = G.
    """
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]] = deque(maxlen=capacity)

    def push_mc(self, s, a, G: float, sT, doneT: float = 1.0, gamma_pow: float = 0.0):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            float(G),
            np.asarray(sT, dtype=np.float32),
            float(doneT),
            float(gamma_pow),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, G, sT, dT, gp = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(G, dtype=np.float32),
            np.asarray(sT, dtype=np.float32),
            np.asarray(dT, dtype=np.float32),
            np.asarray(gp, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------
# Episode collector -> Monte Carlo returns
# ---------------------------
class EpisodeReturnAdder:
    """
    Collects 1-step transitions for ONE episode, then on done=True,
    computes Monte Carlo returns and pushes all steps into main_buffer.

    Stores tmp items:
      (s_t, a_t, r_t, s_{t+1}, done_t)

    On terminal:
      For t from 0..T-1:
        G_t = r_t + gamma r_{t+1} + ... + gamma^{T-1-t} r_{T-1}
      Push (s_t, a_t, G_t, s_T, done=1, gamma_pow=0)
      where s_T is the terminal next_state of the final transition.
    """
    def __init__(self, gamma: float, main_buffer: ReplayBuffer):
        self.gamma = float(gamma)
        self.main_buffer = main_buffer
        self.tmp: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque()

    def reset(self):
        self.tmp.clear()

    def add(self, s, a, r, s2, done: bool):
        self.tmp.append((s, a, float(r), s2, bool(done)))

        if not done:
            return

        # Terminal reached: compute returns backwards
        # s_terminal is the next_state of the last transition
        s_terminal = self.tmp[-1][3]

        G = 0.0
        # iterate backwards over rewards
        for (s_t, a_t, r_t, _s2, _done) in reversed(self.tmp):
            G = r_t + self.gamma * G
            self.main_buffer.push_mc(
                s=s_t,
                a=a_t,
                G=G,
                sT=s_terminal,
                doneT=1.0,
                gamma_pow=0.0
            )

        self.reset()


# ---------------------------
# Config
# ---------------------------
@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    buffer_capacity: int = 100_000
    hidden: Tuple[int, int] = (64, 64)
    activation: Tuple = (nn.ReLU, nn.Tanh)
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0


# ---------------------------
# DDPG Agent (MC-return)
# ---------------------------
class DDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DDPGConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size

        # seeds
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        # networks
        self.actor = ActorNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation).to(self.device)
        self.actor_target = ActorNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation).to(self.device)
        self.critic_target = CriticNetwork(obs_dim, act_dim, cfg.hidden, cfg.activation).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # replay + episode adder
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.mc_adder = EpisodeReturnAdder(gamma=self.gamma, main_buffer=self.buffer)

    @torch.no_grad()
    def choose_action(self, state, noise_std: float = 0.1):
        """
        Returns normalized action in [-1, 1] (assuming actor uses Tanh).
        If your env expects different bounds, scale outside this method.
        """
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).cpu().numpy()[0]
        if noise_std > 0:
            a = a + np.random.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, -1.0, 1.0)

    def store_transition(self, state, action, reward: float, next_state, done: bool):
        """
        Feed 1-step env transition; internally this stores full-episode MC returns when done=True.
        IMPORTANT: this means the replay buffer only grows at episode end.
        """
        self.mc_adder.add(state, action, reward, next_state, done)

    def train(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.buffer) < batch_size:
            return None

        states, actions, Gt, next_states_T, dones_T, gamma_pows = self.buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        Gt = torch.tensor(Gt, dtype=torch.float32, device=self.device).unsqueeze(1)

        # (These are present for compatibility; for MC we do not bootstrap)
        next_states_T = torch.tensor(next_states_T, dtype=torch.float32, device=self.device)
        dones_T = torch.tensor(dones_T, dtype=torch.float32, device=self.device).unsqueeze(1)
        gamma_pows = torch.tensor(gamma_pows, dtype=torch.float32, device=self.device).unsqueeze(1)

        # ----- Critic update (MC target) -----
        # Generic form: y = G + gamma_pow*(1-done)*Q_target(...)
        # For stored MC transitions: gamma_pow=0 and done=1 => y=G
        with torch.no_grad():
            next_actions_T = self.actor_target(next_states_T)
            target_q_T = self.critic_target(next_states_T, next_actions_T)
            y = Gt + gamma_pows * (1.0 - dones_T) * target_q_T  # reduces to Gt

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----- Actor update -----
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- Soft update targets -----
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
        }

    def soft_update(self, net: nn.Module, target_net: nn.Module):
        with torch.no_grad():
            for p, tp in zip(net.parameters(), target_net.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * p.data)

    def save(self, path: str):
        try:
            torch.save({
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "cfg": self.cfg.__dict__,
                "mode": "monte_carlo_returns",
            }, path)
            return f"Model saved successfully at: {path}"
        except Exception as e:
            return f"Error saving model to {path}:: Error: {e}"

    def load(self, path: str, map_location: Optional[str] = None):
        try:
            if map_location is None:
                map_location = self.device

            ckpt = torch.load(path, map_location=map_location, weights_only=False)

            self.actor.load_state_dict(ckpt["actor"])
            self.critic.load_state_dict(ckpt["critic"])
            self.actor_target.load_state_dict(ckpt["actor_target"])
            self.critic_target.load_state_dict(ckpt["critic_target"])
            self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
            self.critic_optimizer.load_state_dict(ckpt["critic_opt"])

            # ensure optimizer tensors are on correct device
            for st in self.actor_optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(self.device)

            for st in self.critic_optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(self.device)

            return f"Model loaded successfully from: {path}"
        except Exception as e:
            return f"Error loading model from {path}: {e}"

    def reset_episode(self):
        """
        Call at episode boundary if you manually manage episodes.
        (Not strictly required if you always pass done=True at terminal.)
        """
        self.mc_adder.reset()
