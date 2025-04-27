'''
Train a RL agent using SAC
'''
import highway_env
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from buffers.prioritized_replay_buffer import PrioritizedReplayBuffer
from buffers.trb import TransformerReplayBuffer

from agents.sactrb import SACTRB
from agents.sactea import SACTEA
from agents.tea_policies import SACTEAPolicy

from agent_eval import *
from highway_config import *
from buffers.trb import *

# print(f'Using highway-env-{highway_env.__version__}')
parser = argparse.ArgumentParser()
parser.add_argument("--train", action='store_true', help="Set true to train SAC agent")
parser.add_argument("--use_trb", action='store_true', help="Set true to use SAC with Transformer Replay Buffer")
parser.add_argument("--use_tea", action='store_true', help="Set true to use SAC with Transformer-Enhanced Actions")
parser.add_argument("--use_per", action='store_true', help="Set true to use SAC with Prioritized Experience Replay")
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
            log_name = 'SACTRB'
            model = SACTRB(
                'MlpPolicy', env,
                batch_size=batch_size, learning_rate=0.0003, learning_starts=learning_starts,
                use_sde=True, use_sde_at_warmup=False, sde_sample_freq=10,
                replay_buffer_class=TransformerReplayBuffer,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        elif use_tea:
            log_name = 'SACTEA'
            model = SACTEA(
                SACTEAPolicy, env,
                batch_size=batch_size, learning_rate=0.0003, learning_starts=learning_starts,
                use_sde=True, use_sde_at_warmup=False, sde_sample_freq=10,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        elif use_per:
            log_name = 'SACPER'
            model = SAC(
                'MlpPolicy', env,
                batch_size=batch_size, learning_rate=0.0003, learning_starts=learning_starts,
                use_sde=True, use_sde_at_warmup=False, sde_sample_freq=10,
                replay_buffer_class=PrioritizedReplayBuffer,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        else:
            log_name = 'SAC'
            model = SAC(
                'MlpPolicy', env,
                batch_size=256, learning_rate=0.0003, learning_starts=learning_starts,
                use_sde=True, use_sde_at_warmup=False, sde_sample_freq=10,
                verbose=0, tensorboard_log=log_path, device=DEVICE
            )

        # ==== Continue training from checkpoint, if desired ====
        # model = SAC.load(f'checkpoints/sac/sac_700K', env)

        for t in range(100, 701, 100):
            model.learn(total_timesteps=100000, progress_bar=True, reset_num_timesteps=False, tb_log_name=log_name)

            if use_trb:
                checkpoint_path = f'checkpoints/critical/sactrb/sactrb_{t}K'
            elif use_tea:
                checkpoint_path = f'checkpoints/critical/sactea/sactea_{t}K'
            elif use_per:
                checkpoint_path = f'checkpoints/critical/sacper/sacper_{t}K'
            else:
                checkpoint_path = f'checkpoints/critical/sac/sac_{t}K'
            model.save(checkpoint_path)  # saves a zip file
            print(f'Saved model checkpoint to: {checkpoint_path}')

    env = make_critical_env()

    if use_trb:
        print('--- SAC+TRB ---')
        model_checkpoint = f'checkpoints/critical/sactrb/sactrb_700K'
        model = SACTRB.load(model_checkpoint)
    elif use_tea:
        print('--- SAC+TEA ---')
        model_checkpoint = f'checkpoints/critical/sactea/sactea_700K'
        env = make_critical_env()
        model.set_parameters(model_checkpoint)  # must pass env to get observation space, required by transformer
    else:
        print('--- SAC Baseline ---')
        model_checkpoint = f'checkpoints/critical/sac/sac_700K'
        model = SAC.load(model_checkpoint)

    get_agent_metrics_parallel(model)
    # render_agent_highway(model, num_episodes=5, record=False, file_prefix='sac')
