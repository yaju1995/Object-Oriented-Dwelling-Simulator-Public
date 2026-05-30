import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from support.lib_config import CustomLogger

logger = CustomLogger(False)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


# --- Critic Network ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))


# --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
        self.position = 0

    def push(self, transition):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size)
        batch = [self.buffer[i] for i in indices]
        return map(np.stack, zip(*batch))

    def __len__(self):
        return len(self.buffer)


class DPGAgent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, seed = 0):

        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        self.name = 'Old Model'
        self.actor = Actor(state_dim, action_dim, max_action=1.0)  # Always output in [-1, 1]
        self.critic = Critic(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer()
        self.gamma = gamma
        self.max_action = max_action
        self.bound_fn = self.default_bound_fn
        self.seed = seed

    def get_action_bounds(self, state, bound_fn=None):
        """
        Compute action bounds based on state and custom logic.

        Parameters:
        - state: array-like, normalized state ∈ [0, 1]
        - bound_fn: callable(state) → (a_min, a_max)
            - if None: use default_bound_fn

        Returns:
        - a_min, a_max: np arrays
        """
        if not callable(bound_fn):
            raise ValueError("bound_fn must be a callable that takes state and returns (a_min, a_max)")

        if bound_fn is None:
            bound_fn = self.default_bound_fn
        return bound_fn(state)

    def default_bound_fn(self, state):
        # Always return static range (no state logic)
        return np.array([-0.5]), np.array([0.5])

    def scale_action(self, raw_action, a_min, a_max):
        return a_min + 0.5 * (raw_action + 1.0) * (a_max - a_min)

    def choose_action(self, state, noise_std=0.1, bound_fn=None):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        raw_action = self.actor(state_tensor).detach().numpy()[0]

        a_min, a_max = self.get_action_bounds(state, bound_fn)
        action = self.scale_action(raw_action, a_min, a_max)

        logger.commandline(f'soc {state[1]}, raw {raw_action}, action {action}, range [{a_min}, {a_max}]')

        noise = np.random.normal(0, noise_std, size=action.shape)
        return np.clip(action + noise, a_min, a_max)

    def store_transition(self, state, action, reward, next_state):
        self.buffer.push((state, action, reward, next_state))

    def train(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states = self.buffer.sample(batch_size)
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)

        # Get dynamic bounds for each next_state (batched)
        a_min_list, a_max_list = [], []
        for ns in next_states.numpy():
            amin, amax = self.get_action_bounds(ns,self.bound_fn)
            a_min_list.append(amin)
            a_max_list.append(amax)
        a_min_tensor = torch.tensor(np.array(a_min_list), dtype=torch.float32)
        a_max_tensor = torch.tensor(np.array(a_max_list), dtype=torch.float32)

        # Actor(next_state) → [-1, 1] → scale to [a_min, a_max]
        raw_next_action = self.actor(next_states)
        next_action = a_min_tensor + 0.5 * (raw_next_action + 1.0) * (a_max_tensor - a_min_tensor)

        with torch.no_grad():
            target_q = rewards + self.gamma * self.critic(next_states, next_action)

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Get dynamic bounds for current states
        a_min_list, a_max_list = [], []
        for s in states.numpy():
            amin, amax = self.get_action_bounds(s,self.bound_fn)
            a_min_list.append(amin)
            a_max_list.append(amax)
        a_min_tensor = torch.tensor(np.array(a_min_list), dtype=torch.float32)
        a_max_tensor = torch.tensor(np.array(a_max_list), dtype=torch.float32)

        raw_action = self.actor(states)
        scaled_action = a_min_tensor + 0.5 * (raw_action + 1.0) * (a_max_tensor - a_min_tensor)

        actor_loss = -self.critic(states, scaled_action).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item())
        }

    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        print(f"Agent saved to {path}")

    def load(self, path):
        checkpoint = torch.load(path,weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"Agent loaded from {path}")
