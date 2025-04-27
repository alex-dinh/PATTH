'''
  Plots:
  - Mean episode reward vs. training steps/epochs
  - Mean episode duration vs. training steps.
  - Average predicted Q-values or V(s) over time (esp for DQN, SAC)
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FONTSIZE = 14
DPI = 150

def thousands_formatter(x, pos):
    return f'{int(x / 1000)}K'

def plot_episode_lengths_sac():
    # File names and corresponding labels
    csv_path = 'data/ep_len_mean/'

    files = {
        csv_path + "SAC_Baseline.csv": "SAC",
        # csv_path + "SACPER.csv": "SAC+PER",
        csv_path + "SACTRB.csv": "SAC+TRB",
        csv_path + "SACTEA.csv": "SAC+TEA"
    }

    plt.figure(figsize=(6, 4))

    for file_name, label in files.items():
        df = pd.read_csv(file_name)
        plt.plot(df["Step"], df["Value"], label=label)

    plt.xlabel("Timestep", fontsize=FONTSIZE)
    plt.ylabel("Mean Episode Length", fontsize=FONTSIZE)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/sac_ep_len_mean.png", dpi=DPI)

def plot_episode_rewards_sac():
    # File names and corresponding labels
    csv_path = 'data/ep_rewards_mean/'

    files = {
        csv_path + "SAC_Baseline.csv": "SAC",
        # csv_path + "SACPER.csv": "SAC+PER",
        csv_path + "SACTRB.csv": "SAC+TRB",
        csv_path + "SACTEA.csv": "SAC+TEA"
    }

    plt.figure(figsize=(6, 4))

    for file_name, label in files.items():
        df = pd.read_csv(file_name)
        plt.plot(df["Step"], df["Value"], label=label)

    plt.xlabel("Timestep", fontsize=FONTSIZE)
    plt.ylabel("Mean Episode Rewards", fontsize=FONTSIZE)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/sac_ep_rewards_mean.png", dpi=DPI)


def plot_episode_lengths_td3():
    # File names and corresponding labels
    csv_path = 'data/ep_len_mean/'

    files = {
        csv_path + "TD3_Baseline.csv": "TD3",
        csv_path + "TD3TRB.csv": "TD3+TRB",
        # csv_path + "TD3TRB.csv": "TD3+TRB",
        csv_path + "TD3TEA.csv": "TD3+TEA"
    }

    plt.figure(figsize=(6, 4))

    for file_name, label in files.items():
        df = pd.read_csv(file_name)
        plt.plot(df["Step"], df["Value"], label=label)

    plt.xlabel("Timestep", fontsize=FONTSIZE)
    plt.ylabel("Mean Episode Length", fontsize=FONTSIZE)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/td3_ep_len_mean.png", dpi=DPI)

def plot_episode_rewards_td3():
    # File names and corresponding labels
    csv_path = 'data/ep_rewards_mean/'

    files = {
        csv_path + "TD3_Baseline.csv": "TD3",
        csv_path + "TD3TRB.csv": "TD3+TRB",
        # csv_path + "TD3PER.csv": "TD3+PER",
        csv_path + "TD3TEA.csv": "TD3+TEA"
    }

    plt.figure(figsize=(6, 4))

    for file_name, label in files.items():
        df = pd.read_csv(file_name)
        plt.plot(df["Step"], df["Value"], label=label)

    plt.xlabel("Timestep", fontsize=FONTSIZE)
    plt.ylabel("Mean Episode Rewards", fontsize=FONTSIZE)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/td3_ep_rewards_mean.png", dpi=DPI)


def plot_episode_lengths_ddpg():
    # File names and corresponding labels
    csv_path = 'data/ep_len_mean/'

    files = {
        csv_path + "DDPG_Baseline.csv": "DDPG",
        csv_path + "DDPGTRB.csv": "DDPG+TRB",
        csv_path + "DDPGTEA.csv": "DDPG+TEA"
    }

    plt.figure(figsize=(6, 4))

    for file_name, label in files.items():
        df = pd.read_csv(file_name)
        plt.plot(df["Step"], df["Value"], label=label)

    plt.xlabel("Timestep", fontsize=FONTSIZE)
    plt.ylabel("Mean Episode Length", fontsize=FONTSIZE)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/ddpg_ep_len_mean.png", dpi=DPI)

def plot_episode_rewards_ddpg():
    # File names and corresponding labels
    csv_path = 'data/ep_rewards_mean/'

    files = {
        csv_path + "DDPG_Baseline.csv": "DDPG",
        csv_path + "DDPGTRB.csv": "DDPG+TRB",
        csv_path + "DDPGTEA.csv": "DDPG+TEA"
    }

    plt.figure(figsize=(6, 4))

    for file_name, label in files.items():
        df = pd.read_csv(file_name)
        plt.plot(df["Step"], df["Value"], label=label)

    plt.xlabel("Timestep", fontsize=FONTSIZE)
    plt.ylabel("Mean Episode Rewards", fontsize=FONTSIZE)
    plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_formatter))
    plt.tick_params(axis='both', which='major', labelsize=FONTSIZE)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/ddpg_ep_rewards_mean.png", dpi=DPI)


plot_episode_lengths_sac()
plot_episode_rewards_sac()
# plot_episode_lengths_td3()
# plot_episode_rewards_td3()
plot_episode_lengths_ddpg()
plot_episode_rewards_ddpg()