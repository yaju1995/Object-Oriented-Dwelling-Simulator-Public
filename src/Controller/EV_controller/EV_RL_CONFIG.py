from Controller.DDPGmodel.DDPG_Agent_multistep import DDPGAgent, DDPGConfig
from torch import nn

EV_DDPG_config = DDPGConfig()
EV_DDPG_config.gamma = 0.9
EV_DDPG_config.actor_lr = 1e-3
EV_DDPG_config.critic_lr = 1e-3
EV_DDPG_config.hidden = (128, 128)
EV_DDPG_config.activation = (nn.ReLU, nn.Sigmoid)
EV_DDPG_config.a_max = 1.0
EV_DDPG_config.a_min = 0.0
EV_DDPG_config.seed = 0
EV_LOOK_AHEAD = 1  # If you update this try to match the input state with this
EV_INPUT_DIM = 3 + EV_LOOK_AHEAD  # try to match the observed state with delay steps
EV_OUT_DIM = 1
EV_MODEL_DIR = f'Models/EV/Trial/n_step_{EV_LOOK_AHEAD}_setup_1/Seed_{EV_DDPG_config.seed}/'
EV_MODEL_NAME = f'states_{EV_INPUT_DIM}_delay_{EV_LOOK_AHEAD}_700eps.pth'


EV_RL_AGENT = DDPGAgent(name='EVagent',
                        obs_dim=EV_INPUT_DIM,
                        act_dim=EV_OUT_DIM,
                        cfg=EV_DDPG_config,
                        n_step=EV_LOOK_AHEAD, #imp
                        return_mode='nstep')

# EV_RL_AGENT = None