import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from Actor import Actor
from Critic import Critic
from ReplayBuffer import ReplayBuffer

class Agent():
    def __init__(self, params):
        self.action_size = params['action_size']
        self.state_size = params['state_size']
        self.num_agents = params['num_agents']
        self.buffer_size = params['buffer_size']
        self.batch_size = params['batch_size']
        self.__gamma = params['gamma']
        self.__tau = params['tau']
        self.__update_every = params['update_every']
        self.__save_to = params['save_to']
        self.__memory = ReplayBuffer(self.buffer_size, self.batch_size)
        self.__lr = params['lr']
        self.noise_type = params['noise_type']

        actor_params = dict()
        actor_params['arch_params_actor'] = params['arch_params_actor']
        actor_params['action_size'] = self.action_size
        actor_params['state_size'] = self.state_size
        actor_params['eps'] = params['eps']
        actor_params['eps_decay'] = params['eps_decay']
        actor_params['eps_min'] = params['min_eps']
        actor_params['noise_type'] = params['noise_type']
        self.actor = Actor(actor_params)
        self.actor_target = Actor(actor_params)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.__lr)
        self.scheduler_actor = optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=100, gamma=0.95)

        critic_params = dict()
        critic_params['arch_params_critic'] = params['arch_params_critic']
        critic_params['action_size'] = self.action_size
        critic_params['state_size'] = self.state_size
        self.critic = Critic(critic_params)
        self.critic_target = Critic(critic_params)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.__lr)
        self.scheduler_critic = optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=100, gamma=0.95)
        self.__t = 0

    def memorize_experience(self, state, action, reward, next_state, done):
        self.__memory.add(state, action.detach(), reward, next_state, done)
        self.__t = (self.__t + 1)

    def learn_from_past_experiences(self):
        if self.__t % self.__update_every == 0:
            if len(self.__memory) > self.batch_size:
                experiences = self.__memory.sample()
                self.update_actor_critic(experiences)

    def choose_action(self, state):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.from_numpy(state.astype(dtype = np.float)).to(device)
        action, action_perturbed = self.actor(state)
        return action, action_perturbed

    def update_actor_critic(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        next_actions, next_actions_perturbed = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.__gamma*Q_targets_next*(1 - dones))  # if done == True: second term is equal to 0
        Q_expected = self.critic(states, actions)
        loss_func = nn.MSELoss()
        loss_critic = loss_func(Q_expected, Q_targets.detach())

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        # self.scheduler_critic.step()
        self.optimizer_critic.step()

        predicted_actions, predicted_actions_perturbed = self.actor(states) # new predicted actions, not the ones stored in buffer

        if self.noise_type == 'parameter':
            #if the distance between predicted_actions and predicted_actions_perturbed is too big (>=0.2) then update noise
            if (predicted_actions-predicted_actions_perturbed).pow(2).mean() >= 0.15:
                self.actor.eps /= 1.01
                self.actor_target.eps /= 1.01
            else:
                self.actor.eps *= 1.01
                self.actor_target.eps *= 1.01

        loss_actor = -self.critic(states, predicted_actions).mean()

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        # self.scheduler_actor.step()
        self.optimizer_actor.step()

        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.actor, self.actor_target)

    def update_eps(self):
        self.actor.eps = max(self.actor.eps*self.actor.eps_decay, self.actor.eps_min)
        self.actor_target.eps = max(self.actor_target.eps*self.actor_target.eps_decay, self.actor_target.eps_min)


    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.__tau * local_param.data + (1.0 - self.__tau) * target_param.data)

    def save_weights(self, save_to):
        actor_params_and_state_dict = {'actor_params': self.actor.actor_params,
                'state_dict': self.actor.state_dict()}
        critic_params_and_state_dict = {'critic_params': self.critic.critic_params,
                'state_dict': self.critic.state_dict()}

        file = dict()
        file['critic_params_and_state_dict'] = critic_params_and_state_dict
        file['actor_params_and_state_dict'] = actor_params_and_state_dict
        torch.save(file, open(save_to, 'wb'))

    def load_weights(self, load_from):
        checkpoint = torch.load(load_from)

        critic_params_and_state_dict = checkpoint['critic_params_and_state_dict']
        actor_params_and_state_dict = checkpoint['actor_params_and_state_dict']

        self.actor = Actor(actor_params_and_state_dict['actor_params'])
        self.actor.load_state_dict(actor_params_and_state_dict['state_dict'])

        self.critic = Critic(critic_params_and_state_dict['critic_params'])
        self.critic.load_state_dict(critic_params_and_state_dict['state_dict'])
        return self


# quick unit test on loading and saving:
# from collections import OrderedDict
# params = dict()
# params['num_agents'] = 2
# params['num_episodes'] = 100        #number of episodes for agent to interact with the environment
# params['action_size'] = 10
# params['state_size'] = 122
# params['buffer_size'] = int(1e6)    # replay buffer size
# params['batch_size'] = 128          # minibatch size
# params['gamma'] = 0.99              # discount factor
# params['tau'] = 1e-2               # for soft update of target parameters
# params['eps'] = 0.3                # exploration factor (modifies noise)
# params['min_eps'] = 0.001           # min level of noise
# min_e = params['min_eps']
# e = params['eps']
# N = params['num_episodes']
# params['eps_decay'] = np.exp(np.log(min_e/e)/(0.8*N)) #decay of the level of the noise after each episode
# params['lr'] = 5e-4                 # learning rate
# params['update_every'] = 2          # how often to update the network (every update_every timestep)
# params['seed'] = np.random.randint(0,100)
# params['max_t'] = 1000              # restriction on max number of timesteps per each episodes
# params['noise_type'] = 'action'     # noise type; can be 'action' or 'parameter'
# params['save_to'] = ('../results/' + 'random_name') # where to save the results to
# params['threshold'] = 600            # the score above which the network parameters are saved
#
# params['arch_params_actor'] = OrderedDict(
#         {'state_and_action_dims': (params['state_size'], params['action_size']),
#          'layers': {
#              # 'Linear_1': 512, 'LayerNorm_1' : None,  'ReLU_1': None,
#              'Linear_2': 256, 'LayerNorm_2' : None,  'ReLU_2': None,
#              'Linear_3': 128, 'LayerNorm_3' : None,  'ReLU_3': None,
#              'Linear_4': 64, 'LayerNorm_4' : None,  'ReLU_4': None,
#              'Linear_5': params['action_size'], 'LayerNorm_5': None,
#              'Tanh_1': None
#          }
#          })
# #parameters for the QNetwork (critic) network
# params['arch_params_critic'] = OrderedDict(
#     {'state_and_action_dims': (params['state_size'], params['action_size']),
#      'layers': {
#          # 'Linear_1': 512, 'ReLU_1': None,
#          'Linear_2': 256, 'ReLU_2': None,
#          'Linear_3': 128, 'ReLU_3': None,
#          'Linear_4': 64,  'ReLU_4': None,
#          'Linear_5': params['action_size']
#      }
#      })
#
# agents = Agents(params)
#
# agents.save_weights(params['save_to'])
# agents.load_weights(params['save_to'])
