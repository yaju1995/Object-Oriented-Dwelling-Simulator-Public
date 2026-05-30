import random
from dataclasses import dataclass
from typing import Deque, Optional, Tuple, Literal
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
                       out_act=activation[1])

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
# Replay Buffer (generic)
# ---------------------------
# ---------------------------
# Replay Buffer (n-step)
# ---------------------------
class ReplayBuffer:
    """
    Stores tuples:
      (s, a, Rn, sN, doneN, gamma_pow)
    where gamma_pow is gamma^k (k steps actually used).
    """
    def __init__(self, capacity: int = 100_000, seed: int = None ):
        self.buf: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, float, float]] = deque(maxlen=capacity)
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def push(self, s, a, Rn: float, sN, doneN: float, gamma_pow: float):
        self.buf.append((
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.float32),
            float(Rn),
            np.asarray(sN, dtype=np.float32),
            float(doneN),
            float(gamma_pow),
        ))

    def sample(self, batch_size: int):
        batch = self.rng.sample(self.buf, batch_size)
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
        self.main_buffer.push(s0, a0, Rn, sN, dN, gp)

        # shift window by one
        self.tmp.popleft()

        # if terminal, flush remaining partial windows
        if done:
            while len(self.tmp) > 0:
                s0, a0, Rn, sN, dN, gp = self._compute()
                self.main_buffer.push(s0, a0, Rn, sN, dN, gp)
                self.tmp.popleft()


# ---------------------------
# MC (episode) builder
# ---------------------------
class EpisodeReturnAdder:
    """
    On episode end, emits MC transitions for each step:
      G_t = r_t + gamma r_{t+1} + ... + gamma^{T-1-t} r_{T-1}
    Stored as:
      (s_t, a_t, G_t, s_T, done=1, gamma_pow=0)
    so the generic target reduces to y = G_t.
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

        s_terminal = self.tmp[-1][3]

        G = 0.0
        for (s_t, a_t, r_t, _s2, _done) in reversed(self.tmp):
            G = r_t + self.gamma * G
            # done=1, gamma_pow=0 => no bootstrap term
            self.main_buffer.push(s_t, a_t, G, s_terminal, 1.0, 0.0)

        self.reset()


# ---------------------------
# One "collector" wrapper to support nstep / mc / hybrid
# ---------------------------
ReturnMode = Literal["nstep", "mc", "hybrid"]


class ReturnCollector:
    """
    mode:
      - "nstep": only n-step transitions
      - "mc": only MC transitions (buffer fills only at episode end)
      - "hybrid": store BOTH (n-step online + MC at episode end)
    """
    def __init__(self, mode: ReturnMode, gamma: float, buffer: ReplayBuffer, n_step: int = 4):
        self.mode = mode
        self.nstep = NStepAdder(n=n_step, gamma=gamma, main_buffer=buffer) if mode in ("nstep", "hybrid") else None
        self.mc = EpisodeReturnAdder(gamma=gamma, main_buffer=buffer) if mode in ("mc", "hybrid") else None

    def reset(self):
        if self.nstep is not None:
            self.nstep.reset()
        if self.mc is not None:
            self.mc.reset()

    def add(self, s, a, r, s2, done: bool):
        if self.nstep is not None:
            self.nstep.add(s, a, r, s2, done)
        if self.mc is not None:
            self.mc.add(s, a, r, s2, done)


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
    hidden: Tuple[int, int] = (128, 128)
    activation: Tuple = (nn.ReLU, nn.Tanh)
    batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: Optional[int] = 0
    a_max: Optional[float] = 1.0
    a_min: Optional[float] = -1.0


# ---------------------------
# DDPG Agent (generic target works for both)
# ---------------------------
class DDPGAgent:
    def __init__(
        self,
        name:str,
        obs_dim: int,
        act_dim: int,
        cfg: DDPGConfig,
        return_mode: ReturnMode = "nstep",
        n_step: int = 4
    ):
        self.name = name
        self.cfg = cfg
        self.device = cfg.device
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.n_step = n_step

        self.rng = np.random.default_rng(cfg.seed)
        # seeds
        random.seed(cfg.seed)
        # np.random.seed(cfg.seed)
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

        # replay + return collector
        self.buffer = ReplayBuffer(cfg.buffer_capacity, seed=cfg.seed)
        self.collector = ReturnCollector(mode=return_mode, gamma=self.gamma, buffer=self.buffer, n_step=n_step)

        self.a_max = self.cfg.a_max
        self.a_min = self.cfg.a_min

    @torch.no_grad()
    def choose_action(self, state, noise_std: float = 0.1):
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(s).cpu().numpy()[0]
        if noise_std > 0:
            a = a + self.rng.normal(0.0, noise_std, size=a.shape)
        return np.clip(a, self.a_min, self.a_max)

    def store_transition(self, state, action, reward: float, next_state, done: bool):
        self.collector.add(state, action, reward, next_state, done)

    def train(self, batch_size: Optional[int] = None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(self.buffer) < batch_size:
            return None
        # print('Updating!')
        states, actions, R, next_states, dones, gamma_pows = self.buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        R = torch.tensor(R, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        gamma_pows = torch.tensor(gamma_pows, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Works for both:
        # - n-step: gamma_pows=gamma^k and dones maybe 0
        # - MC: gamma_pows=0 and dones=1 => y = R (R is G)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = R + gamma_pows * (1.0 - dones) * target_q

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
            "actor_loss": float(actor_loss.item())
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


    def reset_return_builder(self):
        """Call at episode boundary if needed."""
        self.collector.reset()
