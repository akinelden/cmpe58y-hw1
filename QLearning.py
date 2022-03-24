import gym
import numpy as np
import matplotlib.pyplot as plt
import random


class QLearner:
    env: gym.Env
    alpha: float
    gama: float
    epsilon: float
    decay: float
    discrete_state_steps: list[float]
    state_space_limits: list[tuple]  # list of tuple(low, high)
    qtable: np.ndarray
    scores: list[float]

    def __init__(self, env: gym.Env, discrete_space_size: int | list[int], space_limits: list[tuple[int]] = None, alpha: float = 0.1, gama: float = 0.95, epsilon: float = 0.95, decay_rate: float = 0.999):
        self.env = env
        self.alpha = alpha
        self.gama = gama
        self.epsilon = epsilon
        self.decay = decay_rate
        state_space_shape = env.observation_space.shape[0]
        self.set_finite_state_space_limits(space_limits)

        discretization_vector: list[int]  # list of integers olmalı
        if isinstance(discrete_space_size, int):
            discretization_vector = [
                discrete_space_size for _ in range(state_space_shape)]
        else:
            assert len(discrete_space_size) == state_space_shape
            discretization_vector = discrete_space_size.copy()
        self.discrete_state_steps = [
            (h/s - l/s) for (l, h), s in zip(self.state_space_limits, discretization_vector)]
        self.qtable = np.zeros(discretization_vector + [env.action_space.n])
        self.scores = []

    def set_finite_state_space_limits(self, space_limits: list[tuple[int]]):
        if space_limits is None:
            space_limits = [(l, h) for l, h in zip(
                self.env.observation_space.low, self.env.observation_space.high)]
        self.state_space_limits = space_limits

    def find_discrete_state_indices(self, observation: list[float]) -> tuple[int]:
        # since the observation space is continuos, we find the discrete space state indices
        # by dividing current continuos state value to discrete state step size
        indices = [int(o/s - l/s) for o, (l, _h), s in zip(observation,
                                                           self.state_space_limits, self.discrete_state_steps)]
        return tuple(indices)

    def save_qtable(self, filename: str):
        np.save(filename, self.qtable)

    def load_qtable(self, filename: str):
        self.qtable = np.load(filename)

    def run_episode(self, render: bool) -> float:
        score = 0
        done = False
        observation = self.env.reset()
        state = self.find_discrete_state_indices(observation)
        while not done:
            if render:
                self.env.render()
            action: int = -1
            if random.random() > self.epsilon:  # select best action
                q_actions = self.qtable[state]
                # if all actions have same q value select random
                if np.all(q_actions == q_actions[0]):
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.qtable[state])
            else:  # otherwise select random action
                action = self.env.action_space.sample()
            observation, reward, done, _info = self.env.step(action)
            score += reward
            new_state = self.find_discrete_state_indices(observation)
            # Calculate the new Q value for state s and action a -> Qnew = Q + alpha * [ r + gama * max(Q value of s' for all actions of s') - Q ]
            q = self.qtable[state][action]
            q_new = q + self.alpha * \
                (reward + self.gama * np.max(self.qtable[new_state]) - q)
            # update Q value for state s
            self.qtable[state][action] = q_new
            # update current state s
            state = new_state
            # update epsilon value
            self.epsilon *= self.decay
        return score

    def run(self, nEpisodes: int, render: bool = True, log: bool = False):
        self.scores = []
        for i in range(nEpisodes):
            score = self.run_episode(render)
            self.scores.append(score)
            if log:
                print(f"Episode {i} : {score}")
                if i % 10 == 0:
                    print(f"epsilon: {self.epsilon}")

        avg_scores = np.mean(np.array(self.scores[len(self.scores) % 10:]).reshape(-1, 10), axis=1)

        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].plot(self.scores)
        axs[0].set_title("Individual Scores")
        axs[1].plot(avg_scores)
        axs[1].set_title("10-Averaged Scores")
        plt.show()
