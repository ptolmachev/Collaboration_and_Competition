import torch
import numpy as np
from Agent import Agent
from collections import OrderedDict
from unityagents import UnityEnvironment

def perform(Agent, Env, num_agents):
    scores = np.zeros(num_agents)
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations

    actions = []
    for i in range(num_agents):
        action, action_perturbed = Agent.choose_action(states[i])
        actions.append(action_perturbed)
    actions = torch.stack(actions)
    dones = [False] * num_agents

    while not ((np.any(dones) == True)):
        env_info = Env.step(actions.detach().cpu().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        actions = []
        for i in range(num_agents):
            scores[i] += rewards[i]
            action, action_perturbed = Agent.choose_action(states[i])
            actions.append(action_perturbed)
        actions = torch.stack(actions)

        states = next_states
    return scores

### ENVIRONMENT
env_params = {'path' : '../envs/Tennis_Linux/Tennis.x86_64',
          'worker_id' : 0,
          'seed' : np.random.randint(1000),
          'visual_mode' : False,
          'multiagent_mode' : True}
env_name = 'Tennis'
env = UnityEnvironment(file_name=env_params['path'])


#AGENT
params = dict()
brain_name = env.brain_names[0]
params['brain_name'] = brain_name
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]
state_size = len(state)
num_agents = len(env_info.agents)
params['num_agents'] = num_agents
params['num_episodes'] = 1        #number of episodes for agent to interact with the environment
params['action_size'] = action_size
params['state_size'] = state_size
params['buffer_size'] = 10    # replay buffer size
params['batch_size'] = 1          # minibatch size
params['gamma'] = 0.99              # discount factor
params['tau'] = 1e-3               # for soft update of target parameters
params['eps'] = 0.001               # exploration factor (modifies noise)
params['min_eps'] = 0.0         # min level of noise
min_e = params['min_eps']
e = params['eps']
N = params['num_episodes']
params['eps_decay'] = np.exp(np.log(min_e/e)/(0.8*N)) #decay of the level of the noise after each episode
params['lr'] = 1e-3                # learning rate
params['update_every'] = 1          # how often to update the network (every update_every timestep)
params['seed'] = np.random.randint(0,100)
params['noise_type'] = 'action_noise'     # noise type; can be 'action' or 'parameter'
params['save_to'] = ('../results/' + env_name) # where to save the results to
params['threshold'] = 0.5            # the score above which the network parameters are saved
#parameters for the Policy (actor) network

params['arch_params_actor'] = OrderedDict(
        {'state_and_action_dims': (params['state_size'], params['action_size']),
         'layers': {
             # 'Linear_1': 512, 'LayerNorm_1' : None,  'ReLU_1': None,
             # 'Linear_2': 64, 'ReLU_2': None,
             'Linear_3': 128, 'ReLU_3': None,
             'Linear_4': 64, 'ReLU_4': None,
             'Linear_5': params['action_size'],
             'Tanh_1': None
         }
         })
#parameters for the QNetwork (critic) network
params['arch_params_critic'] = OrderedDict(
    {'state_and_action_dims': (params['state_size'], params['action_size']),
     'layers': {
         # 'Linear_1': 512, 'ReLU_1': None,
         # 'Linear_2': 64, 'ReLU_2': None,
         'Linear_3': 128, 'ReLU_3': None,
         'Linear_4': 64,  'ReLU_4': None,
         'Linear_5': params['action_size']
     }
     })

Agent = Agent(params)
load_from = '../results/Tennis_0.8.prms'
Agent.load_weights(load_from)

score = perform(Agent, env,num_agents)
print(score)
