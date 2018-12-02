import torch.nn as nn
import torch
import numpy as np
class Critic(nn.Module):
    def __init__(self, critic_params):
        super(Critic, self).__init__()

        #critic_params
        self.critic_params = critic_params
        self.action_size = critic_params['action_size']
        self.state_size = critic_params['state_size']
        self.arch_params = critic_params['arch_params_critic']
        torch.manual_seed(np.random.randint(100))
        list_of_layers = []
        keys = list(self.arch_params['layers'].keys())
        prev_layer_size = self.state_size+self.action_size
        for i in range(len(self.arch_params['layers'])):
            key = keys[i]
            layer_type = key.split('_')[0]
            if layer_type == 'Linear':
                layer_size = self.arch_params['layers'][key]
                list_of_layers.append(nn.Linear(prev_layer_size, layer_size))
                prev_layer_size = layer_size
            elif layer_type == 'LayerNorm':
                list_of_layers.append(nn.LayerNorm(prev_layer_size))
            elif layer_type == 'ReLU':
                list_of_layers.append(nn.ReLU())
            elif layer_type == 'Tanh':
                list_of_layers.append(nn.Tanh())
            else:
                print("Error: got unspecified layer type: '{}'. Check your layers!".format(layer_type))
                break
        self.layers = nn.ModuleList(list_of_layers)

    def forward(self, state, action):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        state = torch.tensor(state).float()
        action = torch.tensor(action).float()
        x = torch.cat((state, action), dim = 1).to(device)
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


#Quick unit test:
# from collections import OrderedDict
# import numpy as np
# params = dict()
# params['num_agents'] = 2
# params['num_episodes'] = 100        #number of episodes for agent to interact with the environment
# params['action_size'] = 5
# params['state_size'] = 25
# params['buffer_size'] = int(1e6)    # replay buffer size
# params['batch_size'] = 128          # minibatch size
# params['gamma'] = 0.99              # discount factor
# #parameters for the QNetwork (critic) network
# params['arch_params_critic'] = OrderedDict(
#     {'state_and_action_dims': (params['state_size'], params['action_size']),
#      'layers': {
#          'Linear_1': 512, 'ReLU_1': None,
#          'Linear_2': 256, 'ReLU_2': None,
#          'Linear_3': 128, 'ReLU_3': None,
#          'Linear_4': 64,  'ReLU_4': None,
#          'Linear_5': 1
#      }
#      })
#
# Q = Critic(params)
# # state = torch.from_numpy(np.random.rand(params['state_size'])).unsqueeze(0)
# # action = torch.from_numpy(np.random.rand(params['action_size'])).unsqueeze(0)
# state = torch.from_numpy(np.random.rand(10,params['state_size']))
# action = torch.from_numpy(np.random.rand(10,params['action_size']))
#
# print(Q(state, action))
# print('done')