"""
DQN with n-step returns (discrete action spaces).

Target:
  y = G_n + (gamma^n) * (1 - done_n) * max_a' Q_target(s_n, a')

If n_step=1 => standard DQN.
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
    def __init__(self, obs_dim: int, n_actions: int, hidden=(128, 128), activation=nn.ReLU):
        super().__init__()
        self.net = mlp([obs_dim, *hidden, n_actions], activation=activation, out_act=nn.Identity)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------
# Replay Buffer (n-step)
# ---------------------------
class ReplayBufferNStep:
    """
    Stores tuples:
      (s, a, G_n, s_n, done_n, gamma_pow)

    gamma_pow will be gamma^n (or gamma^k if episode ends early within window).
    """
    def __init__(self, capacity: int = 100_000):
        self.buf: Deque[Tuple[np.ndarray, int, float, np.ndarray, float, float]] = deque(maxlen=capacity)

    def push(self, s, a: int, G_n: float, s_n, done_n: bool, gamma_pow: float):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            int(a),
            float(G_n),
            np.asarray(s_n, dtype=np.float32),
            float(done_n),
            float(gamma_pow),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, Gn, sN, dN, gp = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(Gn, dtype=np.float32),
            np.asarray(sN, dtype=np.float32),
            np.asarray(dN, dtype=np.float32),
            np.asarray(gp, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buf)


class NStepAdder:
    """
    Converts 1-step transitions into n-step transitions.

    Input each env step:
      (s_t, a_t, r_t, s_{t+1}, done_t)

    Produces:
      (s_t, a_t, G_n, s_{t+n}, done_n, gamma^n)
    where if terminal occurs before n steps, it truncates:
      done_n = 1, s_{t+k} is terminal next_state,
      gamma_pow = gamma^k, and G_n sums only available rewards.
    """
    def __init__(self, gamma: float, n_step: int, main_buffer: ReplayBufferNStep):
        assert n_step >= 1
        self.gamma = float(gamma)
        self.n_step = int(n_step)
        self.main_buffer = main_buffer
        self.tmp: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()

    def reset(self):
        self.tmp.clear()

    def _flush_all(self):
        """
        Called at episode end: create remaining truncated n-step transitions.
        """
        while len(self.tmp) > 0:
            self._emit_one()
            self.tmp.popleft()

    def _emit_one(self):
        """
        Emit n-step transition for the oldest element in tmp, using up to n steps.
        """
        s0, a0, *_ = self.tmp[0]

        G = 0.0
        gamma_pow = 1.0
        done_n = False
        s_n = None

        # accumulate rewards up to n or until done
        for k in range(min(self.n_step, len(self.tmp))):
            s_k, a_k, r_k, s_k1, d_k = self.tmp[k]
            G += gamma_pow * float(r_k)
            gamma_pow *= self.gamma
            s_n = s_k1
            if d_k:
                done_n = True
                break

        # If we used k+1 steps, gamma_pow is gamma^(k+1) (or gamma^n if full).
        self.main_buffer.push(
            s=s0,
            a=a0,
            G_n=G,
            s_n=s_n,
            done_n=done_n,
            gamma_pow=gamma_pow
        )

    def add(self, s, a: int, r: float, s2, done: bool):
        self.tmp.append((s, int(a), float(r), s2, bool(done)))

        # If we have at least n steps, emit one n-step transition
        if len(self.tmp) >= self.n_step:
            self._emit_one()
            self.tmp.popleft()

        # If episode ended, flush remaining truncated transitions
        if done:
            self._flush_all()
            self.reset()


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

    # n-step
    n_step: int = 3

    # epsilon-greedy
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000

    # target update
    target_update_every: int = 1000  # steps

    # misc
    grad_clip_norm: Optional[float] = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0


# ---------------------------
# DQN Agent (n-step)
# ---------------------------
class DQNAgent:
    def __init__(self,name, obs_dim: int, n_actions: int, cfg: DQNConfig):
        self.name = name
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = float(cfg.gamma)
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

        self.optimizer = optim.Adam(self.q.parameters(), lr=cfg.lr)

        # replay + n-step adder
        self.buffer = ReplayBufferNStep(cfg.buffer_capacity)
        self.nstep_adder = NStepAdder(gamma=self.gamma, n_step=cfg.n_step, main_buffer=self.buffer)

        self.step_count = 0

    def epsilon(self) -> float:
        if self.cfg.eps_decay_steps <= 0:
            return self.cfg.eps_end
        frac = min(1.0, self.step_count / float(self.cfg.eps_decay_steps))
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    @torch.no_grad()
    def choose_action(self, state, greedy: bool = False) -> int:
        self.step_count += 1
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.n_actions)

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q_values = self.q(s)
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state, action: int, reward: float, next_state, done: bool):
        """
        Feed 1-step env transition; internally creates n-step transitions.
        Buffer grows throughout the episode (unlike your MC episodic version).
        """
        self.nstep_adder.add(state, action, reward, next_state, done)

    def train(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.buffer) < batch_size:
            return None

        states, actions, G_n, next_states_n, dones_n, gamma_pows = self.buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)  # [B,1]
        G_n = torch.tensor(G_n, dtype=torch.float32, device=self.device).unsqueeze(1)        # [B,1]
        next_states_n = torch.tensor(next_states_n, dtype=torch.float32, device=self.device)
        dones_n = torch.tensor(dones_n, dtype=torch.float32, device=self.device).unsqueeze(1)
        gamma_pows = torch.tensor(gamma_pows, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s,a)
        q_sa = self.q(states).gather(1, actions)

        # target: G_n + gamma^n * (1-done_n) * max_a' Q_target(s_n, a')
        with torch.no_grad():
            q_next_max = self.q_target(next_states_n).max(dim=1, keepdim=True)[0]
            y = G_n + gamma_pows * (1.0 - dones_n) * q_next_max

        loss = nn.MSELoss()(q_sa, y)

        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip_norm)
        self.optimizer.step()

        # periodic hard update
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
                "mode": "dqn_nstep",
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

            # move optimizer tensors to device
            for st in self.optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(self.device)

            return f"Model loaded successfully from: {path}"
        except Exception as e:
            return f"Error loading model from {path}: {e}"

    def reset_episode(self):
        """
        Call at episode boundary if you manually manage episodes.
        Not required if you always pass done=True correctly.
        """
        self.nstep_adder.reset()