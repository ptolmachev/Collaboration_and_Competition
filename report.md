Skip to content
 
Search or jump to…

Pull requests
Issues
Marketplace
Explore
 @ptolmachev Sign out
1
0 0 ptolmachev/Collaboration_and_Competition
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Insights  Settings
Collaboration_and_Competition/ 
README.md
  or cancel
 Indent mode Indent size Line wrap mode
1
# Collaboration_and_Competition
2
​
3
Udacity Reinforcment learning online course project 3 solution and the **report**.
4
​
5
### Project description
6
This is an implementation of self-playing DDPG (Deep Deterministic Policy Gradient) algorithm applied to solve the specified below task.
7
​
8
In this environment, called Tennis, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
9
​
10
The observation space consists of 24 variables corresponding to the position and velocity of the ball and racket (8 variables) of the three last frames. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.
11
​
12
**Solution criteria**
13
The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents).
14
After each episode, one adda up the rewards that each agent received (without discounting), to get a score for each agent. 
15
This yields 2 (potentially different) scores. One then takes the maximum of these 2 scores.
16
This yields a single score for each episode.
17
**The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5**.
18
​
19
### Environment visualisation 
20
​
21
<p align="center">
22
<img src="https://github.com/ptolmachev/Collaboration_and_Competition/blob/master/img/Tennis_performance.gif"/>
23
</p>
24
​
25
​
26
### Environment setup
27
Install Unity3d:
28
https://unity3d.com/get-unity/download
29
​
30
For detailed Python environment setup (PyTorch, the ML-Agents toolkit, and a few more Python packages) please follow these steps:
31
[link](https://github.com/udacity/deep-reinforcement-learning#dependencies)
32
​
33
Download pre-built Unity Environment:
34
  - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
35
  - [Mac](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
36
  - [Win x32](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
37
  - [Win x64](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
38
​
39
### Theoretical background
40
The DDPG algorithm was firstly presented in the papaer [Lillicrap et. al](https://arxiv.org/abs/1509.02971).
41
The pseudocode for this algorithm can be summarised as following:
42
<p align="center">
@ptolmachev
Commit changes
Commit summary 
Update README.md
Optional extended description

Add an optional extended description…
  Commit directly to the master branch.
  Create a new branch for this commit and start a pull request. Learn more about pull requests.
 
© 2018 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
Pricing
API
Training
Blog
About
Press h to open a hovercard with more details.
