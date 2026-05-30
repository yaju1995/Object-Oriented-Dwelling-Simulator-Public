
from Controller.DQNmodel.DQN_Agent_multistep import DQNConfig, DQNAgent

HVAC_DQN_config  = DQNConfig()
HVAC_DQN_config.gamma = 0.95
HVAC_DQN_config.lr = 1e-4
HVAC_DQN_config.hidden = (512,256)
HVAC_DQN_config.seed = 0
HVAC_DQN_config.batch_size = 1000
HVAC_DQN_config.n_step = 4
HVAC_DQN_config.eps_decay_steps = 20_000
HVAC_LOOK_AHEAD = HVAC_DQN_config.n_step
HVAC_INPUT_DIM = 2 + HVAC_LOOK_AHEAD*2
HVAC_OUT_DIM = 2
HVAC_MODEL_DIR = f'Models/HVAC/test/NSTEP_{HVAC_LOOK_AHEAD}_bound_2_setup4_reward-3-Test8'
HVAC_MODEL_NAME = f'states_{HVAC_INPUT_DIM}_delay_{HVAC_LOOK_AHEAD}_500eps.pth'

# HVAC_RL_AGENT = DQNAgent(name='HVACagent', obs_dim=HVAC_INPUT_DIM, n_actions=HVAC_OUT_DIM,cfg=HVAC_DQN_config)
HVAC_RL_AGENT = None
action_map = {
    0: 0,
    1: 1
}