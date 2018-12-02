import torch.nn as nn
from Ornstein_Uhlenbeck_Noise import Ornstein_Uhlenbeck_Noise
import torch
import numpy as np
class Actor(nn.Module):
    def __init__(self, actor_params):
        super(Actor, self).__init__()

        #Actor_params
        self.actor_params = actor_params
        self.arch_params = actor_params['arch_params_actor']
        self.action_size = actor_params['action_size']
        self.state_size = actor_params['state_size']
        self.eps = actor_params['eps']
        self.eps_decay = actor_params['eps_decay']
        self.eps_min = actor_params['eps_min']
        self.noise_type = actor_params['noise_type']
        torch.manual_seed(np.random.randint(100))
        if self.noise_type == 'action_noise':
            self.action_noise = Ornstein_Uhlenbeck_Noise(self.action_size)
        else:
            pass


        list_of_layers = []
        prev_layer_size = self.state_size
        keys = list(self.arch_params['layers'].keys())
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



    def forward(self, state):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        x = torch.tensor(state).float().to(device)
        if self.noise_type == 'action_noise':
            for i in range(len(self.layers)):
                x = self.layers[i](x)
            x_perturbed = torch.clamp((x + self.eps*torch.from_numpy(self.action_noise.generate_noise()).float()),-1,1)
            return x, x_perturbed

        elif self.noise_type == 'parameter_noise':
            pass
        else:
            raise KeyError('Got an unspecified type of noise')



# Quick unit test
# from collections import OrderedDict
# import numpy as np
# params = dict()
# params['action_size'] = 2
# params['state_size'] = 25          # discount fActor
# params['eps'] = 0.1
# params['eps_decay'] = 0.99
# params['eps_min'] = 0.001
# params['noise_type'] = 'action_noise'
# #parameters for the QNetwork (critic) network
# params['arch_params_actor'] = OrderedDict(
#     {'state_and_action_dims': (params['state_size'], params['action_size']),
#      'layers': {
#          'Linear_1': 512, 'ReLU_1': None,
#          'Linear_2': 256, 'ReLU_2': None,
#          'Linear_3': 128, 'ReLU_3': None,
#          'Linear_4': 64,  'ReLU_4': None,
#          'Linear_5': params['action_size']
#      }
#      })
#
# A = Actor(params)
# state = torch.from_numpy(np.random.rand(params['state_size'])).unsqueeze(0)
# # state = torch.from_numpy(np.random.rand(10,params['state_size']))
#
# print(A(state))
# print('done')