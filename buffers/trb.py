import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np



class TransformerReplayBuffer(ReplayBuffer):
    def __init__(self, buffer_size, observation_space, action_space, device="auto", n_envs=1,
                 optimize_memory_usage=False, lr=3e-4):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs,
                         optimize_memory_usage=optimize_memory_usage)
        self.obs_dim = int(np.prod(observation_space.shape))
        self.next_obs_dim = int(np.prod(observation_space.shape))
        self.action_dim = action_space.shape[0]

        self.attention_model = ReplayTransformer(self.obs_dim, self.action_dim, embed_dim=128, num_heads=4, device=self.device)
        self.optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=lr)
        self.env_idx = 0  # if using vec_env, will be randomized every attention network update
        self.preds = None
        self.sample_indices = None

    def sample(self, batch_size, env: Optional[VecNormalize] = None):
        self.env_idx = torch.randint(0, high=self.n_envs, size=(1,)).item()
        # Randomly sample N * batch_size indices
        upsample_size = int(batch_size * 2)
        # Must be cpu since replay_data is initially numpy arrays
        if self.full:
            random_inds = (torch.randint(1, self.buffer_size, size=(upsample_size,), device='cpu')
                           + self.pos) % self.buffer_size
        else:
            random_inds = torch.randint(0, self.pos, size=(upsample_size,), device='cpu')

        # Resample indices using attention-weighted probabilities, narrow down to batch_size
        weights = self.get_attention_weights(random_inds)
        random_inds = random_inds.to(self.device)
        indices = torch.multinomial(weights, batch_size, replacement=False)
        self.sample_indices = indices
        final_inds = random_inds[indices].detach().cpu().numpy()
        return self._get_samples(final_inds, env)

    def get_attention_weights(self, batch_inds) -> torch.Tensor:
        """
        Compute attention weights for a batch of transitions.
        Args:
            batch: Tensor of shape (batch_size, obs_dim + action_dim + 1)
        Returns:
            weights: Tensor of shape (batch_size,)
        """
        # Convert buffer replay data to tensors
        obs = torch.tensor(self.observations[batch_inds, self.env_idx], device=self.device)
        next_obs = torch.tensor(self.next_observations[batch_inds, self.env_idx], device=self.device)
        obs_flat = obs.reshape(obs.shape[0], -1)
        next_obs_flat = next_obs.reshape(next_obs.shape[0], -1)
        actions = torch.tensor(self.actions[batch_inds, self.env_idx], device=self.device)
        rewards = torch.tensor(self.rewards[batch_inds, self.env_idx], device=self.device).unsqueeze(1)

        # Forward pass through attn network, use to predict TD errors
        pred_td_errors = self.attention_model.forward(obs_flat, next_obs_flat, actions, rewards).squeeze(1)  # Shape: [batch_size]
        self.preds = pred_td_errors

        # mps softmax NaN bug workaround
        weights = F.log_softmax(pred_td_errors.abs(), dim=0)
        weights = torch.exp(weights)
        return weights

    def transformer_loss(self, td_errors):
        # loss = F.l1_loss(self.preds[self.sample_indices], td_errors, reduction='sum')
        loss = F.mse_loss(self.preds[self.sample_indices], td_errors, reduction='sum')
        return loss

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, self.env_idx, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, self.env_idx, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, self.env_idx, :], env),
            self.actions[batch_inds, self.env_idx, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, self.env_idx] * (1 - self.timeouts[batch_inds, self.env_idx])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, self.env_idx].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class ReplayTransformer(nn.Module):
    def __init__(self, obs_dim, action_dim, embed_dim=128, num_heads=4, device='cpu'):
        super().__init__()
        self.feature_dim = embed_dim
        self.device = device
        self.embed = nn.Linear(obs_dim * 2 + action_dim + 1, self.feature_dim, device=self.device)

        self.attn = nn.MultiheadAttention(self.feature_dim, num_heads, batch_first=True, device=self.device)

        self.ffn = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        ).to(self.device)

        self.score_proj = nn.Linear(self.feature_dim, 1, device=self.device)

    def forward(self, obs_flat, next_obs_flat, actions, rewards):
        inputs = torch.cat([obs_flat, next_obs_flat, actions, rewards], dim=-1)
        embeddings = self.embed(inputs)  # use same embeddings for Q, K, V
        attn_out, _ = self.attn.forward(embeddings, embeddings, embeddings)
        ffn_out = self.ffn(attn_out)
        scores = self.score_proj.forward(ffn_out)
        return scores
