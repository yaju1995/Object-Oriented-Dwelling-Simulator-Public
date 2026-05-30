"""
DDPG with n-step returns (multi-step reward) for continuous action spaces.

Key idea:
  y = R_n + (gamma^k) * (1 - done_n) * Q_target(s_{t+k}, pi_target(s_{t+k}))
where:
  R_n = r_t + gamma r_{t+1} + ... + gamma^{k-1} r_{t+k-1}
  k = n unless terminal happens earlier (then k < n)

This file includes:
- MLP helper
- Actor / Critic networks
- ReplayBuffer that stores n-step transitions + gamma^k
- NStepAdder that converts 1-step env transitions into n-step stored transitions
- DDPGConfig
- DDPGAgent (choose_action, store_transition, train, save, load)
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
                       out_act=activation[1])  # final activation clamps to [-1,1] for Tanh

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
# Replay Buffer (n-step)
# ---------------------------
class ReplayBuffer:
    """
    Stores tuples:
      (s, a, Rn, sN, doneN, gamma_pow)
    where gamma_pow is gamma^k (k steps actually used).
    """
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]] = deque(maxlen=capacity)

    def push_nstep(self, s, a, Rn: float, sN, doneN: float, gamma_pow: float):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            float(Rn),
            np.asarray(sN, dtype=np.float32),
            float(doneN),
            float(gamma_pow),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, Rn, sN, dN, gp = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            np.asarray(Rn, dtype=np.float32),
            np.asarray(sN, dtype=np.float32),
            np.asarray(dN, dtype=np.float32),
            np.asarray(gp, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


# ---------------------------
# n-step transition builder
# ---------------------------
class NStepAdder:
    """
    Collects 1-step transitions, emits n-step transitions into main_buffer.

    tmp stores up to n items of (s, a, r, s2, done).
    When enough steps collected (or episode ends), it computes:
      s0, a0,
      Rn = sum_{i=0..k-1} gamma^i r_i
      sN = s_{t+k}
      doneN = done encountered within k steps (0 or 1)
      gamma_pow = gamma^k
    """
    def __init__(self, n: int, gamma: float, main_buffer: ReplayBuffer):
        assert n >= 1
        self.n = n
        self.gamma = gamma
        self.main_buffer = main_buffer
        self.tmp: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=n)

    def reset(self):
        self.tmp.clear()

    def _compute(self):
        R = 0.0
        gamma_pow = 1.0

        s0, a0 = self.tmp[0][0], self.tmp[0][1]

        # build R until terminal or until we consumed current tmp contents
        s_last, done_last = None, 0.0
        for (_, _, r, s2, done) in self.tmp:
            R += gamma_pow * float(r)
            s_last = s2
            done_last = float(done)
            if done:
                break
            gamma_pow *= self.gamma

        # gamma_pow is gamma^k, where k is number of rewards used
        return s0, a0, R, s_last, done_last, gamma_pow

    def add(self, s, a, r, s2, done: bool):
        self.tmp.append((s, a, r, s2, done))

        # if we don't yet have n steps and not terminal, wait for more
        if len(self.tmp) < self.n and not done:
            return

        # emit one n-step transition starting at tmp[0]
        s0, a0, Rn, sN, dN, gp = self._compute()
        self.main_buffer.push_nstep(s0, a0, Rn, sN, dN, gp)

        # shift window by one
        self.tmp.popleft()

        # if terminal, flush remaining partial windows
        if done:
            while len(self.tmp) > 0:
                s0, a0, Rn, sN, dN, gp = self._compute()
                self.main_buffer.push_nstep(s0, a0, Rn, sN, dN, gp)
                self.tmp.popleft()


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
# DDPG Agent (n-step)
# ---------------------------
class DDPGAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DDPGConfig, n_step: int = 4):
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.n_step = n_step

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

        # replay + n-step builder
        self.buffer = ReplayBuffer(cfg.buffer_capacity)
        self.nstep_adder = NStepAdder(n=n_step, gamma=self.gamma, main_buffer=self.buffer)

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
        return np.clip(a, -0.0, 1.0)

    def store_transition(self, state, action, reward: float, next_state, done: bool):
        """
        Feed 1-step env transition; internally this stores n-step transitions.
        """
        self.nstep_adder.add(state, action, reward, next_state, done)

    def train(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.buffer) < batch_size:
            return None  # nothing to train yet

        states, actions, Rn, next_states, dones, gamma_pows = self.buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        Rn = torch.tensor(Rn, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        gamma_pows = torch.tensor(gamma_pows, dtype=torch.float32, device=self.device).unsqueeze(1)

        # ----- Critic update (n-step TD target) -----
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            # y = Rn + gamma^k * (1 - done) * Q_target(s_{t+k}, a_{t+k})
            y = Rn + gamma_pows * (1.0 - dones) * target_q

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
                "n_step": self.n_step,
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

            # restore n_step with check
            if "n_step" in ckpt:
                if ckpt["n_step"] != self.n_step:
                    return f"Warning: n_step mismatch (ckpt={ckpt['n_step']}, current={self.n_step})"
                self.n_step = ckpt["n_step"]

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

    def reset_nstep(self):
        """
        Call at episode boundary if you manually manage episodes.
        (Not strictly required if you always pass done=True at terminal.)
        """
        self.nstep_adder.reset()
