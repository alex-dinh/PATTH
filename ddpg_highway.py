'''
Train a RL agent using DDPG.

DDPG can be considered a special case of TD3 that uses 1 critic and no policy update delaying.
'''

import highway_env
import argparse
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from buffers.trb import TransformerReplayBuffer

from agents.td3trb import TD3TRB
from agents.td3tea import TD3TEA
from agents.tea_policies import TD3TEAPolicy

from agent_eval import *
from highway_config import *
from buffers.trb import *

# print(f'Using highway-env-{highway_env.__version__}')
parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="Set true to train a DDPG agent")
parser.add_argument("--use_trb", action='store_true', help="Set true to use DDPG with Transformer Replay Buffer")
parser.add_argument("--use_tea", action='store_true', help="Set true to use DDPG with Transformer-Enhanced Actions")
parser.add_argument("--use_per", action='store_true', help="Set true to use DDPG with Prioritized Experience Replay")
parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()
# ==================================
#            Main script
# ==================================
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
train = args.train
use_trb = args.use_trb
use_tea = args.use_tea
use_per = args.use_per
num_envs = args.num_envs

if __name__ == '__main__':
    n_cpu = num_envs
    if train:
        if args.debug:  # for single thread debugging
            env = make_critical_env()
            learning_starts = 64
            batch_size = 64
            log_path = None
        else:
            env = make_vec_env(make_critical_env, n_envs=n_cpu, seed=0, vec_env_cls=SubprocVecEnv)
            learning_starts = 10000
            batch_size = 256
            log_path = 'tb_logs/'  # path for tensorboard logging
            # log_path = None

        if use_trb:
            log_name = 'DDPGTRB'
            model = TD3TRB(
                'MlpPolicy', env,
                batch_size=batch_size, learning_rate=1e-3, learning_starts=learning_starts,
                replay_buffer_class=TransformerReplayBuffer, policy_kwargs={"n_critics": 1},
                policy_delay=1, target_noise_clip=0.0, target_policy_noise=0.1,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        elif use_tea:
            log_name = 'DDPGTEA'
            model = TD3TEA(
                TD3TEAPolicy, env,
                batch_size=batch_size, learning_rate=1e-3, learning_starts=learning_starts,
                policy_kwargs={"n_critics": 1},
                policy_delay=1, target_noise_clip=0.0, target_policy_noise=0.1,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        elif use_per:
            log_name = 'DDPGPER'
            model = TD3(
                'MlpPolicy', env,
                batch_size=batch_size, learning_rate=1e-3, learning_starts=learning_starts,
                replay_buffer_class=PrioritizedReplayBuffer, policy_kwargs={"n_critics": 1},
                policy_delay=1, target_noise_clip=0.0, target_policy_noise=0.1,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        else:
            log_name = 'DDPG'
            model = TD3(
                'MlpPolicy', env,
                batch_size=256, learning_rate=1e-3, learning_starts=learning_starts,
                policy_kwargs={"n_critics": 1},
                policy_delay=1, target_noise_clip=0.0, target_policy_noise=0.1,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        for t in range(100, 701, 100):
            model.learn(total_timesteps=100000, progress_bar=True, reset_num_timesteps=False, tb_log_name=log_name)

            if use_trb:
                checkpoint_path = f'checkpoints/critical/ddpgtrb/ddpgtrb_{t}K'
            elif use_tea:
                checkpoint_path = f'checkpoints/critical/ddpgtea/ddpgtea_{t}K'
            elif use_per:
                checkpoint_path = f'checkpoints/critical/ddpgper/ddpgper_{t}K'
            else:
                checkpoint_path = f'checkpoints/critical/ddpg/ddpg_{t}K'
            model.save(checkpoint_path)  # saves a zip file
            print(f'Saved model checkpoint to: {checkpoint_path}')


    if args.use_trb:
        print('--- DDPG+TRB ---')
        # model_checkpoint = f'checkpoints/critical/ddpgtrb/ddpgtrb_400K'
        # model = TD3TRB.load(model_checkpoint)
    elif args.use_tea:
        print('--- DDPG+TEA ---')
        model_checkpoint = f'checkpoints/critical/ddpgtea/ddpgtea_700K'
        # model_checkpoint = f'checkpoints/critical/td3tea/td3tea_rs0.2_final'
        env = make_critical_env()
        model = TD3TEA(TD3TEAPolicy, env, policy_kwargs={"n_critics": 1})
        model.set_parameters(model_checkpoint)
    else:
        print('--- DDPG Baseline ---')
        model_checkpoint = f'checkpoints/critical/ddpg/ddpg_700K'
        model = TD3.load(model_checkpoint)


    get_agent_metrics_parallel(model)
    # render_agent_highway(model, num_episodes=5, record=False, file_prefix='ddpg')
