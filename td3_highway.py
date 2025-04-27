'''
Train a RL agent using TD3
'''

import highway_env
import argparse
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
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
parser.add_argument("--train", action='store_true', help="Set true to train TD3 agent")
parser.add_argument("--use_trb", action='store_true', help="Set true to use TD3 with Transformer Replay Buffer")
parser.add_argument("--use_tea", action='store_true', help="Set true to use TD3 with Transformer-Enhanced Actions")
parser.add_argument("--use_per", action='store_true', help="Set true to use TD3 with Prioritized Experience Replay")
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
            log_name = 'TD3TRB'
            model = TD3TRB(
                'MlpPolicy', env,
                batch_size=batch_size, learning_rate=1e-3, learning_starts=learning_starts,
                replay_buffer_class=TransformerReplayBuffer,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        elif use_tea:
            log_name = 'TD3TEA'
            model = TD3TEA(
                TD3TEAPolicy, env,
                batch_size=batch_size, learning_rate=1e-3, learning_starts=learning_starts,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        elif use_per:
            log_name = 'TD3PER'
            model = TD3(
                'MlpPolicy', env,
                batch_size=batch_size, learning_rate=1e-3, learning_starts=learning_starts,
                replay_buffer_class=PrioritizedReplayBuffer,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        else:
            log_name = 'TD3'
            model = TD3(
                'MlpPolicy', env,
                batch_size=256, learning_rate=1e-3, learning_starts=learning_starts,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        for t in range(100, 701, 100):
            model.learn(total_timesteps=100000, progress_bar=True, reset_num_timesteps=False, tb_log_name=log_name)

            if use_trb:
                checkpoint_path = f'checkpoints/critical/td3trb/td3trb_{t}K'
            elif use_tea:
                checkpoint_path = f'checkpoints/critical/td3tea/td3tea_{t}K'
            elif use_per:
                checkpoint_path = f'checkpoints/critical/td3per/td3per_{t}K'
            else:
                checkpoint_path = f'checkpoints/critical/td3/td3_{t}K'
            model.save(checkpoint_path)  # saves a zip file
            print(f'Saved model checkpoint to: {checkpoint_path}')


    if args.use_trb:
        print('--- TD3+TRB ---')
        model_checkpoint = f'checkpoints/critical/td3trb/td3trb_700K'
        model = TD3TRB.load(model_checkpoint)
    if args.use_tea:
        print('--- TD3+TEA ---')
        model_checkpoint = f'checkpoints/critical/td3tea/td3tea_700K'
        env = make_critical_env()
        model = TD3TEA(TD3TEAPolicy, env, verbose=0)
        model.set_parameters(model_checkpoint)
    else:
        print('--- TD3 Baseline ---')
        model_checkpoint = f'checkpoints/critical/td3/td3_700K'
        model = TD3.load(model_checkpoint)


    get_agent_metrics_parallel(model)
    # render_agent_highway(model, num_episodes=5, record=False, file_prefix='td3trb/td3trb')
