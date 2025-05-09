# Pay Attention to the Highway
This project introduces and implements two transformer methods for improving deep reinforcement learning algorithms: the Transformer Replay Buffer and Transformer-Enhanced Actions. Agents are trained under an environment simulated by the highway-env package using gymnasium.

*This is a collaboration between Alexander Dinh, Gian Iacona, Shawn Mishra, and Michael Mendoza at Georgia Tech for the Spring 2025 Deep Learning course.*

## Setup
```commandline
git clone https://github.com/alex-dinh/PATTH
cd PATTH
pip install -r requirements.txt
cd HighwayEnv
pip install -e .
```

## Usage and Options

### DDPG
```commandline
python ddpg_highway.py [--train] [--use_trb | --use_tea] [--debug]
```

### TD3
```commandline
python td3_highway.py [--train] [--use_trb | --use_tea] [--debug]
```

### SAC
```commandline
python sac_highway.py [--train] [--use_trb | --use_tea] [--debug]
```

Notes:
- `--train`: Enables training mode. 
- `--use_trb`: Use the Transformer Replay Buffer
- `--use_tea`: Use Transformer-Enhanced Actions
- If neither `--use_trb` nor `--use_tea` is specified, the baseline algorithm with a standard replay buffer is used.
- `--debug`: For use with `--train`. Forces single-threaded execution to allow for use with debugger. By default, training uses 8 vectorized (parallel) environments.

## Demos
### SAC+TRB
![sactrb](https://github.com/user-attachments/assets/306ee85a-4cd1-4604-961f-576525d0b593)

### SAC+TEA
![sactea](https://github.com/user-attachments/assets/28f466ec-ebfa-4992-96d5-3985b324afe8)

