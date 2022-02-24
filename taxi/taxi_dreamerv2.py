
#importing gym and dreamer v2 api
import gym
import dreamerv2.api as dreamer_v2
import gym_minigrid
from visgrid.taxi.taxi_gym_env import TaxiEnv
import os, shutil

from matplotlib import pyplot as plt

#for debugging
import pdb



#update the configurations for training
config_data = {

    # train script
    'logdir':'./logs',
    'seed': 0,
    'task':'taxi_env',
    'envs': 1,
    'envs_parallel': 'none',
    'render_size': [120,120],
    'dmc_camera': -1,
    'atari_grayscale': False, #for atari suite, converts to grayscale image
    'time_limit': 2000,
    'action_repeat': 1, #multiplier when logging action
    'steps': 700, #max number of steps to take
    'log_every': 50, #number of steps to log video and metrics
    'eval_every': 50, #evaluate policy every 'k' steps: Actor-Critic
    'eval_eps': 1, #number of episodes to run evaluation?
    'prefill': 200, #number of steps (in 1 epoch) to 'prefill' before world model sequencing [learning world model] [random agent acts for 'k' steps]
    'pretrain': 1,
    'train_every': 1,
    'train_steps': 25,
    'expl_until': 5, #number of steps to initially explore
    'replay': {'capacity': 400, 'ongoing': True, 'minlen': 3, 'maxlen': 25, 'prioritize_ends': True},
    'dataset': {'batch': 16, 'length': 40},
    'log_keys_video': ['image'],
    'log_keys_sum': '^$',
    'log_keys_mean': '^$',
    'log_keys_max': '^$',
    'precision': 16,
    'jit': True,

    # agent
    'clip_rewards': 'tanh',
    'expl_behavior': 'greedy',
    'expl_noise': 0.0,
    'eval_noise': 0.0,
    'eval_state_mean': False,

    # world model
    'grad_heads': ['decoder', 'reward', 'discount'],
    'pred_discount': True,
    'rssm': {'ensemble': 1, 'hidden': 1024, 'deter': 1024, 'stoch': 32, 'discrete': 32, 'act': 'elu', 'norm': 'none', 'std_act': 'sigmoid2', 'min_std': 0.1},
    'encoder': {'mlp_keys': '.*', 'cnn_keys': '.*', 'act': 'elu', 'norm': 'none', 'cnn_depth': 48, 'cnn_kernels': [4, 4, 4, 4], 'mlp_layers': [400, 400, 400, 400]},
    'decoder': {'mlp_keys': '.*', 'cnn_keys': '.*', 'act': 'elu', 'norm': 'none', 'cnn_depth': 48, 'cnn_kernels': [5, 5, 6, 6], 'mlp_layers': [400, 400, 400, 400]},
    'reward_head': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'mse'},
    'discount_head': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'binary'},
    'loss_scales': {'kl': 1.0, 'reward': 1.0, 'discount': 1.0, 'proprio': 1.0},
    'kl': {'free': 0.0, 'forward': False, 'balance': 0.8, 'free_avg': True},
    'model_opt': {'opt': 'adam', 'lr': 1e-4, 'eps': 1e-5, 'clip': 100, 'wd': 1e-6},

    # actor critic
    'actor': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'auto', 'min_std': 0.1},
    'critic': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'mse'},
    'actor_opt': {'opt': 'adam', 'lr': 4e-5, 'eps': 1e-5, 'clip': 100, 'wd': 1e-6},
    'critic_opt': {'opt': 'adam', 'lr': 2e-4, 'eps': 1e-5, 'clip': 100, 'wd': 1e-6},
    'discount': 0.99,
    'discount_lambda': 0.95,
    'imag_horizon': 5,
    'actor_grad': 'auto',
    'actor_grad_mix': 0.1,
    'actor_ent': 1e-3,
    'slow_target': True,
    'slow_target_update': 100,
    'slow_target_fraction': 1,
    'slow_baseline': True,
    'reward_norm': {'momentum': 1.0, 'scale': 1.0, 'eps': 1e-8},

    # exploration
    'expl_intr_scale': 1.0,
    'expl_extr_scale': 0.0,
    'expl_opt': {'opt': 'adam', 'lr': 3e-4, 'eps': 1e-5, 'clip': 100, 'wd': 1e-6},
    'expl_head': {'layers': 4, 'units': 400, 'act': 'elu', 'norm': 'none', 'dist': 'mse'},
    'expl_reward_norm': {'momentum': 1.0, 'scale': 1.0, 'eps': 1e-8},
    'disag_target': 'stoch',
    'disag_log': False,
    'disag_models': 10,
    'disag_offset': 1,
    'disag_action_cond': True,
    'expl_model_loss': 'kl'

}

configurations = dreamer_v2.defaults.update(config_data).parse_flags()

env = TaxiEnv()
pdb.set_trace()
env = common.GymWrapper(env)
env = common.ResizeImage(env)
if hasattr(env.act_space['action'], 'n'):
    env = common.OneHotAction(env)
else:
    env = common.NormalizeAction(env)
env = common.TimeLimit(env, config.time_limit)

pdb.set_trace()

dreamer_v2.train(env, configurations)
