import numpy as np
import torch
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from highway_config import make_critical_env


def get_agent_metrics(env, model, device='cuda'):
    num_crashes = 0
    num_episodes = 100
    total_rewards = 0
    total_ep_len = 0
    for ep in tqdm(range(num_episodes)):
        obs, info = env.reset()
        done = truncated = False
        ep_rewards = 0
        ep_len = 0
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            # env.render()
            ep_rewards += reward
            ep_len += 1
            if done:
                num_crashes += 1
        total_rewards += ep_rewards
        total_ep_len += ep_len
        # print(f"Episode: {ep + 1}, Rewards: {ep_rewards}, Episode Length: {ep_len}, Crashes: {num_crashes}")
    print(f"Episodes: {num_episodes}\n"
          f"Avg Rewards: {total_rewards / num_episodes}\n"
          f"Avg Episode Len: {total_ep_len / num_episodes}\n"
          f"Crashes: {num_crashes}")
    env.close()


def get_agent_metrics_parallel(model, vec_env=None, num_envs=8):
    if vec_env is None:
        vec_env = make_vec_env(make_critical_env, n_envs=num_envs, seed=0, vec_env_cls=SubprocVecEnv)

    if not isinstance(vec_env, SubprocVecEnv):
        raise Exception('Must pass VecEnv for parallel processing.')

    # Switch to eval/test mode (this affects batch norm / dropout)
    model.policy.set_training_mode(False)

    # Initialize tracking variables
    num_episodes = 200

    num_crashes = 0
    episode_rewards = []
    episode_lengths = []
    env_rewards = np.zeros(num_envs)
    env_episode_lengths = np.zeros(num_envs)
    episode_count = np.zeros(num_envs)

    # Reset all environments
    obs = vec_env.reset()

    progress_bar = tqdm(total=num_episodes)
    total_completed_episodes = 0

    while total_completed_episodes < num_episodes:
        # Convert observations to tensor and get batch predictions
        obs_tensor = torch.FloatTensor(obs).cpu()
        with torch.no_grad():
            # For SAC, use nondeterministic predictions. Has a stochastic actor by design
            # Does not matter for DDPG/TD3
            actions, _state = model.predict(obs_tensor, deterministic=True)

        # Step environments forward with the actions
        obs, rewards, dones, infos = vec_env.step(actions)

        # Render all environments simultaneously (opens multiple windows)
        # vec_env.render()

        # Update tracking variables
        env_rewards += rewards
        env_episode_lengths += 1

        # Handle completed episodes
        for i in range(num_envs):
            if dones[i]:
                episode_count[i] += 1
                if infos[i]["TimeLimit.truncated"] is False:  # If terminated (not truncated), count as crash
                    num_crashes += 1

                # Update progress bar for completed episodes
                total_completed_episodes += 1
                progress_bar.update(1)

                # Update totals and reset reward and length tracking for this env
                episode_rewards.append(env_rewards[i])
                env_rewards[i] = 0
                episode_lengths.append(env_episode_lengths[i])
                env_episode_lengths[i] = 0

                if total_completed_episodes >= num_episodes:
                    break

    # Calculate metrics
    print(f"\nEpisodes: {total_completed_episodes}\n"
          f"Avg Rewards: {np.mean(episode_rewards)}\n"
          f"Avg Episode Len: {np.mean(episode_lengths)}\n"
          f"Crashes: {num_crashes}")


def render_agent_highway(model, env=None, num_episodes=5, record=False, file_prefix='highway-video'):
    if env is None:
        env = make_critical_env()

    if record:  # set True to save videos
        env = RecordVideo(env, video_folder='videos', episode_trigger=lambda e: True, name_prefix=file_prefix)
        env.unwrapped.set_record_video_wrapper(env)

    env.render()
    num_crashes = 0
    for ep in range(num_episodes):
        obs, _info = env.reset()
        terminated = truncated = False
        ep_rewards = 0
        ep_len = 0
        while not (terminated or truncated):
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            # print(action, reward)
            env.render()
            ep_rewards += reward
            ep_len += 1
            if terminated:
                num_crashes += 1
                break
        print(f"Episode: {ep + 1}, Rewards: {ep_rewards}, Episode Length: {ep_len}, Crashes: {num_crashes}")
    env.close()
