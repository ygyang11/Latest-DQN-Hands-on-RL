# Latest-DQN-Hands-on-RL
based on the website 《动手学强化学习》, where the codes was several years ago and the called gym lib has already been out of date. Thus, somewhere need to be changed, here is the complete new code on DQN and other variants.
If you just want to implement the DQN, you can just download it. 
**However, the parts that need to be changed on the other algorithmd  using gym in hands_on_RL is the same.  so here we show the parts need to be changed to help you change the other algorithms**
## Need to be changed
### 1. The environment ‘env_name-v0’ is out of date. You should consider upgrading to version `v1`
```
env_name = 'CartPole-v1'
env = gym.make(env_name, render_mode='human')
```

### 2. env.seed(0) is not used, you should changed to there, and add '[0]' behind
```
state = env.reset(seed = 0)[0]
```

### 3. too many values to unpack (expected 4):next_state, reward, done, _ = env.step(action)
Add a return value
```
next_state, reward, done, _, __ = env.step(action)
```

### 4. Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() 
```
def take_action(self. state)
    ....
    'state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)'
    ....
```

### 5. the new version env cant stop automaticly, so we need to add a counter
```
....
    for i_episode in range(int(num_episodes / 10)):
        episode_return = 0
        state = env.reset(seed=0)[0]
        done = False
        'steps = 0'
        while not done 'and steps < 200':
            'steps += 1'
            ....
```
