# Collaboration_and_Competition

Udacity Reinforcment learning online course project 3 solution.

### Project description
This is an implementation of self-playing DDPG (Deep Deterministic Policy Gradient) algorithm applied to solve the specified below task.

In this environment, called Tennis, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket (8 variables) of the three last frames. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

**Solution criteria**
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).
After each episode, one adda up the rewards that each agent received (without discounting), to get a score for each agent. 
This yields 2 (potentially different) scores. One then takes the maximum of these 2 scores.
This yields a single score for each episode.
**The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5**.

### Environment visualisation (untrained agent)
<p align="center">
<https://github.com/ptolmachev/Collaboration_and_Competition/blob/master/img/Tennis_performance.gif"/>
</p>

### Environment setup
Install Unity3d:
https://unity3d.com/get-unity/download

For detailed Python environment setup (PyTorch, the ML-Agents toolkit, and a few more Python packages) please follow these steps:
[link](https://github.com/udacity/deep-reinforcement-learning#dependencies)

Download pre-built Unity Environment:
  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  - [Win x32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  - [Win x64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

### Theoretical background
The DDPG algorithm was firstly presented in the papaer [Lillicrap et. al](https://arxiv.org/abs/1509.02971).
The pseudocode for this algorithm can be summarised as following:
<p align="center">
<img src="https://github.com/ptolmachev/Continuous_Control/blob/master/img/DDPG_algorithm.jpg"/>
</p>

The idea behind the algorithm:

Given the state of an Agent in the Environment, the Policy network returns an action from the continuous action space slightly perturbed by noise for the exploration purposes. 

The QNetwork then evaluates this action given the state (So the networks accepts concatenated vector state-action and returns a single value).

The corresponding implementation of abovementioned algorithm is stated below with the auxiliary comments:

```python
    def update_Qnet_and_policy(self, experiences):
        states, actions, rewards, next_states, dones = experiences 
        # ^ sample random experiences from the memory
        
        next_actions, next_actions_perturbed = self.actor_target(next_states) 
        # ^ get actions from the next states according to target policy
        
        Q_targets_next = self.critic_target(next_states, next_actions) 
        # ^ evaluate the q-function for the next states and next actions
        
        Q_targets = rewards + (self.__gamma*Q_targets_next*(1 - dones))  
        # ^ get target q-function value for the current states and actions
        
        Q_expected = self.critic_local(states, actions)
        # ^ get the estimation of q-function value for the current states and actions according 
        # to critic network
        
        loss_func = nn.MSELoss()
        loss_critic = loss_func(Q_expected, Q_targets.detach())
        # ^ define the loss functions for critic

        self.optimizer_critic.zero_grad()
        loss_critic.backward()
        self.optimizer_critic.step()
        # ^ update the critic network (q-network)

        predicted_actions, predicted_actions_perturbed = self.actor_local(states) 
        # ^ get new predicted actions, not the ones stored in buffer

        loss_actor = -self.critic_local(states, predicted_actions).mean()
         # ^ define the loss functions for actor

        self.optimizer_actor.zero_grad()
        loss_actor.backward()
        self.optimizer_actor.step()
        # ^ update the actor network (policy)
```

To solve the task one self-playing Agent was initialized. The Agent then decided for every racket which action to take (based on local observations):

```python
  for e in range(num_episodes):
      scores = np.zeros(num_agents)
      # get the initial state
      env_info = env.reset(train_mode=True)[brain_name]
      states = env_info.vector_observations
      # get the initial action
      actions = []
      for i in range(num_agents):
          action, action_perturbed = Agent.choose_action(states[i])
          actions.append(action_perturbed)
      actions = torch.stack(actions)
      dones = [False] * num_agents
      
      # while episode is not finished do following get next states, rewards and dones
      # for the agents and chose actionsfor both agents after 
      while not ((np.any(dones) == True)):
          env_info = Env.step(actions.detach().cpu().numpy())[brain_name]
          next_states = env_info.vector_observations
          rewards = env_info.rewards
          dones = env_info.local_done
          #get next actions for both agents
          actions = []
          for i in range(num_agents):
              scores[i] += rewards[i]
              action, action_perturbed = Agent.choose_action(states[i])
              actions.append(action_perturbed)
              Agent.memorize_experience(states[i], action_perturbed, rewards[i], next_states[i], dones[i])
          actions = torch.stack(actions)
          states = next_states
          # self-explanatory
          Agent.learn_from_past_experiences()
        # decrease an exploration parameter
        Agent.update_eps()
        current_score = np.max(scores)
        scores_window.append(current_score)
        scores_log.append(current_score)
```

### Hyperparameters

The following hyperparameters were used:

```python
params['num_agents'] = num_agents   # 2
params['num_episodes'] = 5000       #number of episodes for agent to interact with the environment
params['action_size'] = action_size # 2
params['state_size'] = state_size   # 24
params['buffer_size'] = int(1e5)    # replay buffer size
params['batch_size'] = 128          # minibatch size
params['gamma'] = 0.99              # discount factor
params['tau'] = 1e-2                # for soft update of target parameters
params['eps'] = 0.5                 # exploration factor (modifies noise)
params['min_eps'] = 0.005           # min level of noise
min_e = params['min_eps']           
e = params['eps']
N = params['num_episodes']
params['eps_decay'] = np.exp(np.log(min_e/e)/(0.8*N)) #decay of the level of the noise after each episode
params['lr'] = 5e-4                # learning rate
params['update_every'] = 1         # how often to update the network (every update_every timestep)
params['seed'] = np.random.randint(0,100)
params['noise_type'] = 'action_noise'     # noise type; can be 'action_noise' or 'parameter_noise'
params['save_to'] = ('../results/' + env_name) # where to save the results to
params['threshold'] = 0.5            # the score above which the network parameters are saved
```
parameters for Ornstein-Uhlenbeck noise were set to: mu = 0, sigma = 0.2, theta = 0.15

**The network architecture for actor**:

| Layer   | (in, out)          | Activation|
|---------|--------------------|-----------|
| Layer 1 | (32, `action_size`)| `linear`  |
| Layer 2 | (`state_size`, 128) | `relu`   |
| Layer 3 | (32, `action_size`)| `linear`  |
| Layer 4 | (128, 32)          | `relu`    |
| Layer 5 | (32, `action_size`)| `linear`  |
| Layer 6 | (`action_size`, `action_size`) | `tanh` |

+ Orstein-Uhlenbeck noise of the output, then the entries in the action vector wer clipped to fit [-1 1] range.

**The network architecture for critic**:

| Layer   | (in, out)          | Activation|
|---------|--------------------|-----------|
| Layer 1 | (32, `action_size`)| `linear`  |
| Layer 2 | (`state_size`, 128) | `relu`   |
| Layer 3 | (32, `action_size`)| `linear`  |
| Layer 4 | (128, 32)          | `relu`    |
| Layer 5 | (32, `action_size`)| `linear`  |



### Code organisation
The implementation is stored in the folder 'src', which includes:
- `interact_and_train.py`- the main file, to run the training of reinforcment learning agent. It includes hyperparameters and fucntion 'interact_and_train' which makes created Agent and Environment to interact.
- `Agent.py` - contains the implementation of an agent.
- `ReplayBuffer.py` - implementation of internal buffer to sample the experiences from it.
- `Ornstein_Uhlenbeck_Noise.py` - contains class to generate correlated noise (random walk)
- `Critic.py` - an ANN to evaluate the value of the undertaken action.
- `Actor.py` - an ANN to choose action.
- `plotter.py` - generates the plot of scores acquired during the training.
- `perform.py` - Initializes an agent with specified state dictionary and architecture loaded from file and then runs visualization of the agent's performance.

The weight of the network of trained agent are located in 'results' directory

### Performance of a trained agent
Here is the plot of an Agent's performacne, which achieved the 0.5 score in 2962 episodes:

![scores](https://github.com/ptolmachev/Collaboration_and_Competition/blob/master/img/res4.png)

### Suggested improvements
There a many possible venues of boosting the algorithm's performance:
- Reduce the state vector by the factor of 2 in size by extracting parameters of the ball from the corresponding parameters of the racket (unfortunately, I couldn't do this, because there is no explanation which parameters in the state vector are which) 
- Try other strategies like:
    - Initializing two completely separated agents (both having their own Actor and Critic networks)
    - Initializing Agents with the same Critic network but utilising different Actor networks
- It may be of worth to make Replay Buffer to sample the experiences from the memory based on their priority (Prioritised Experience Replay)
- Parameter noise seems to perform better if implemented correctly (see https://blog.openai.com/better-exploration-with-parameter-noise/)
- There is always a room to tune some hyperparamters!
