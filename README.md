# Tidy Reinforcement Learning with Tensorflow
I am a researcher working on automation tasks using deep reinforcement learning. The paper and the reality were quite different, and there was a lot of difficulty during the automation task. I create this repository to help those who start a task using deep reinforcement learning. All of the code is in tensorflow and Python 3.


## Objective
* The code in this repo has a simple and pretty structure in which each algorithm achieves uniformity. It will be a huge help for you to understand the differences between the reinforcement learning algorithms.
* Take advantage of the pseudo code folder. You should be able to see pseudo code and make all the algorithms into a consistent architecture.


## List of Implemented Algorithms
We used open-gym cartpoles (for discrete tasks), mountain car (for continous tasks), and pendulum (for continous tasks). In the case of HER, we used the coin flipping environment cited in the paper.

* Policy Gradient (PG) for cartpole (discrete task)
* Advantage Actor-Critic (A2C) for cartpole (discrete task)
* Advantage Actor-Critic (A2C) for mountain car (continuous task),,,(imperfect...!!!)
* Proximal Policy Optimization (PPO) (continuous task)
* Deep Q Network (DQN) for cartpole (discrete task)
* Deep Deterministic Policy Gradient (DDPG) for pendulum (continuous task)
* Hindsight Experience Replay for coin flipping (discrete task),,,(imperfect...!!!)
* Soft Actor-Critic (SAC) for pendulum (continuous task)


## Papers / Pseudo Codes of RL Algorithms
* Policy Gradient (PG) [[Paper]](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf) [[Pseudo Code]](./Pseudo_code/PG.png)
* Proximal Policy Optimization (PPO) [[Paper]](https://arxiv.org/abs/1707.06347) [[Pseudo Code]](./Pseudo_code/PPO.png)
* Asynchronous Advantage Actor-Critic (A3C) [[Paper]](https://arxiv.org/abs/1602.01783) [[Pseudo Code]](./Pseudo_code/A3C.png)
* Deep Q-learning Network (DQN) [[Paper]](https://arxiv.org/abs/1312.5602) [[Pseudo Code]](./Pseudo_code/DQN.png)
* Deep Deterministic Policy Gradient (DDPG) [[Paper]](https://arxiv.org/abs/1509.02971) [[Pseudo Code]](./Pseudo_code/DDPG.png)
* Hindsight Experience Replay (HER) [[Paper]](https://arxiv.org/abs/1707.01495) [[Pseudo Code]](./Pseudo_code/HER.png)
* Soft Actor-Critic (SAC) [[Paper]](https://arxiv.org/abs/1801.01290) [[Pseudo Code]](./Pseudo_code/SAC.png)


## Compare the Following Algorithms
* PG (cartpole) vs A2C (cartpole)
* A2C (cartpole) vs A2C (pendulum)
* A2C (pendulum) vs PPO (pendulum)
* PG (cartpole) vs DQN (cartpole)
* A2C (pendulum) vs DDPG (pendulum)
* DQN (cartpole) vs HER (coin)
* PPO (pendulum) vs SAC (pendulum)


## Some Tips in Realistic Development
* A positive reward is a magnet, and a negative reward is a mole game.
* The fastest agent to learn the Pendulum task is DDPG. So is ddpg the best reinforcement learning algorithm?
* Which of the sparse reward and dense reward is practical?
* If the problem is difficult, split it up and approach it.
* Without a simulator, there is no answer other than model-based running.
* In very difficulut task, it does not affect the skill drawn from experience replay.
* Hindsight Experience Replay changes reward function. Unless you are a master of reinforcement learning, do not try HER.


## Installation
```
pip install -r requirements.txt
```

## Inspired by
* [OpenAI - Spinning Up](https://github.com/openai/spinningup)
* [Reinforcement Learning Methods and Tutorials](https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow)
* [OpenAI - Baselines](https://github.com/openai/baselines)
