#run the environment
import torch
from collections import OrderedDict, deque
import numpy as np
from unityagents import UnityEnvironment
from Agent import Agent
from plotter import plotter
import pickle


def nice_print(score, scores_window, eps, e):
    print('\rEpisode {}\t Average Score: {:.6f}\t Current Score : {}\t epsilon :{:.4f}'.
          format(e + 1, np.mean(scores_window), score, eps),end="")
    if (e + 1) % 100 == 0:
        print('\rEpisode {}\tAverage Score: {:.6f}'.format(e + 1, np.mean(scores_window)))

def save_best_score(scores_window, best_score, threshold, save_to, e):
    if (np.mean(scores_window) >= threshold and (np.mean(scores_window) > best_score)):
        best_score = np.mean(scores_window)
        print('\nEnvironment achieved average score {:.6f} in {:d} episodes!'.format(np.mean(scores_window), (e + 1)))
        file_name = str(save_to) + '_' + str(np.round(np.mean(scores_window), 2)) + str('.prms')
        Agent.save_weights(str(file_name))
        print("environment saved to ", file_name)
        return best_score
    else:
        if best_score < np.mean(scores_window):
            return np.mean(scores_window)
        else:
            return best_score

def interact_and_train(Agent, Env, params):

    num_episodes = params['num_episodes']
    num_agents = params['num_agents']
    brain_name = Env.brain_names[0]
    save_to = params['save_to']
    threshold = params['threshold']
    scores_log = []
    scores_window = deque(maxlen = 100)
    best_score = -np.inf
    for e in range(num_episodes):

        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations

        actions = []
        for i in range(num_agents):
            action, action_perturbed = Agent.choose_action(states[i])
            actions.append(action_perturbed)
        actions = torch.stack(actions)
        dones = [False] * num_agents
        ####

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
                Agent.memorize_experience(states[i], action_perturbed, rewards[i], next_states[i], dones[i])
            actions = torch.stack(actions)

            states = next_states
            Agent.learn_from_past_experiences()
        ####

        Agent.update_eps()
        current_score = np.max(scores)
        scores_window.append(current_score)
        scores_log.append(current_score)

        nice_print(current_score, scores_window, Agent.actor.eps, e)
        best_score = save_best_score(scores_window, best_score, threshold, save_to, e)

    return scores_log


env_params = {'path' : '../envs/Tennis_Linux/Tennis.x86_64',
          'worker_id' : 0,
          'seed' : np.random.randint(1000),
          'visual_mode' : False,
          'multiagent_mode' : True}
env_name = 'Tennis'
env = UnityEnvironment(file_name=env_params['path'])

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
params['num_episodes'] = 5000        #number of episodes for agent to interact with the environment
params['action_size'] = action_size
params['state_size'] = state_size
params['buffer_size'] = int(1e5)    # replay buffer size
params['batch_size'] = 128          # minibatch size
params['gamma'] = 0.99              # discount factor
params['tau'] = 1e-2               # for soft update of target parameters
params['eps'] = 0.5               # exploration factor (modifies noise)
params['min_eps'] = 0.005           # min level of noise
min_e = params['min_eps']
e = params['eps']
N = params['num_episodes']
params['eps_decay'] = np.exp(np.log(min_e/e)/(0.8*N)) #decay of the level of the noise after each episode
params['lr'] = 5e-4                # learning rate
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
             'Linear_3': 100, 'ReLU_3': None,
             'Linear_4': 32, 'ReLU_4': None,
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
         'Linear_3': 100, 'ReLU_3': None,
         'Linear_4': 32,  'ReLU_4': None,
         'Linear_5': params['action_size']
     }
     })

Agent = Agent(params)
scores = interact_and_train(Agent, env, params)

pickle.dump(scores, open('../results/' + env_name + '.pkl', 'wb+'))
scores = pickle.load(open('../results/' + env_name + '.pkl', 'rb+'))
plotter(scores,0.5, filt = True)