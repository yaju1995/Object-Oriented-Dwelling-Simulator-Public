from Controller.DDPGmodel.bounded_DDPG_Agent_multistep import DDPGConfig
from Controller.DDPGmodel.bounded_DDPG_Agent_multistep import Bound_DDPGAgent
from torch import nn

ESS_DDPG_config = DDPGConfig()
ESS_DDPG_config.gamma = 1
ESS_DDPG_config.actor_lr = 1e-3
ESS_DDPG_config.critic_lr = 1e-3
ESS_DDPG_config.hidden = (256,128,)  # need to keep this fixed
ESS_DDPG_config.activation = (nn.ReLU, nn.Tanh)
ESS_DDPG_config.batch_size = 1000
ESS_DDPG_config.a_max = 1.0
ESS_DDPG_config.a_min = -1.0
ESS_DDPG_config.n_step = 4
# ESS_DDPG_config.tau = 1
ESS_DDPG_config.seed = 4
ESS_LOOK_AHEAD = ESS_DDPG_config.n_step  # If you update this try to match the input state with this
ESS_INPUT_DIM = 2 + ESS_LOOK_AHEAD*2  # try to match the observed state with delay steps
ESS_OUT_DIM = 1


ESS_MODEL_DIR = f'./Models/ESS/Train_5kWh_0_5C/SAFE_DDPG_N_step_{ESS_DDPG_config.n_step}Y_100_RES_60mins_SEED{ESS_DDPG_config.seed}'
# ESS_MODEL_DIR = f'../Models/ESS/old FW_60minsC'
ESS_MODEL_NAME = f'seed_{ESS_DDPG_config.seed}.pth'




ESS_RL_AGENT = Bound_DDPGAgent(name='ESSagent',
                               obs_dim=ESS_INPUT_DIM,
                               act_dim=ESS_OUT_DIM,
                               cfg=ESS_DDPG_config,
                               return_mode='nstep')

# ESS_RL_AGENT = None
