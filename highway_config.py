import gymnasium as gym


def make_configure_env(**kwargs):
    env = gym.make(kwargs["id"], config=kwargs["config"], render_mode="rgb_array")
    env = env.unwrapped
    env.reset()
    return env


def make_critical_env():
    env = gym.make("critical-v0", render_mode="rgb_array")
    env = env.unwrapped
    env.reset()
    return env


def make_critical_tenv_hard():
    env = gym.make("critical-v0", render_mode="rgb_array")
    env = env.unwrapped
    env.config["aggressive_vehicle_ratio"] = 0.2
    env.config["defensive_vehicle_ratio"] = 0.2
    env.config["truck_vehicle_ratio"] = 0.1
    env.reset()
    return env


def make_critical_discrete_env():
    config = {
        "action": {
            "type": "DiscreteMetaAction",  # 5 possible actions
        },
        "duration": 50,  # total timesteps = duration * policy_frequency
        "policy_frequency": 1,  # number of times agent performs action per second
        "simulation_frequency": 5,
        "lanes_count": 3,
        "lane_change_reward": 0,
        'vehicles_count': 25,
    }
    env = gym.make("critical-v0", config=config, render_mode="rgb_array")
    env = env.unwrapped
    env.reset()
    return env

