import numpy as np
import gym

env = gym.make("MountainCar-v0")
env.reset()

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

q_table = np.random.uniform(low=-2, high=0)


# done = False
#
# while not done:
#     # 0 = left, 1 = nothing ,2 = right
#     action = 2
#     new_state, reward, done, _ = env.step(action)
#     print(new_state, reward, done)
#     env.render()
#
# env.close()
