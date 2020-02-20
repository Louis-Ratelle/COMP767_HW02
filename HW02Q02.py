import numpy as np
import gym
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tiling

SEED = None
GAMMA = 1
EPSILON = 0.1
TILINGS = 8
SIZE = 4096
RUNS = 20
EPISODES = 1000
MAX_STEPS = 2000
ENV = 'MountainCar-v0'

TRAINING_EPISODES = 5
TESTING_EPISODES = 5
TEST_EVERY = 10
TOL = 1e-6

# #############################################################################
#
# Parser
#
# #############################################################################


def get_arguments():
    def _str_to_bool(s):
        '''Convert string to boolean (in argparse context)'''
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(
        description='Applying dynamic programming '
        'to a discrete gym environment.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('--env', type=str, default=ENV,
                        help="The environment to be used. Default:"
                        + ENV)
    parser.add_argument('--gamma', type=float, default=GAMMA,
                        help='Defines the discount rate. Default: '
                        + str(GAMMA))
    parser.add_argument('--epsilon', type=float, default=EPSILON,
                        help='Defines the parameter epsilon for the '
                        'epsilon-greedy policy. Default: '
                        + str(EPSILON))
    parser.add_argument('-t', '--tilings', type=int, default=TILINGS,
                        help='Number of runs to be executed. Default: '
                        + str(TILINGS))
    parser.add_argument('-s', '--size', type=int, default=SIZE,
                        help='Size of index hash table of the tiling algorithm. ''
                        'Default: ' + str(SIZE))
    parser.add_argument('-n', '--runs', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(RUNS))
    parser.add_argument('--episodes', type=int,
                        default=EPISODES,
                        help='Number of episodes to be executed. '
                        'Default: ' + str(EPISODES))
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS,
                        help='Number of maximum steps allowed in a single '
                        'episode. Default: ' + str(MAX_STEPS))
    parser.add_argument('-v', '--verbose', action="store_true",
                        help='If this flag is set, the algorithm will generate '
                        'more output, useful for debugging.')

    parser.add_argument('--tol', type=float, default=TOL,
                        help='Defines the tolerance for convergence of the '
                        'algorithms. Default: ' + str(TOL))
    parser.add_argument('--training_episodes', type=int,
                        default=TRAINING_EPISODES,
                        help='Number of training episodes to be executed.'
                        'Default: ' + str(TRAINING_EPISODES))
    parser.add_argument('--testing_episodes', type=int, default=TESTING_EPISODES,
                        help='Number of testing episodes to be executed.'
                        'Default: ' + str(TESTING_EPISODES))
    parser.add_argument('--test_every', type=int, default=TEST_EVERY,
                        help='Number of training iterations to execute before '
                        'each test. Default: ' + str(TEST_EVERY))

    return parser.parse_args()


# #############################################################################
#
# Plotting
#
# #############################################################################


def plot2(title, cumulative_reward_3d, timesteps_3d):
    '''Creates the two required plots: cumulative_reward and number of timesteps
    per episode.'''

    fig, axs = plt.subplots(nrows=1, ncols=2,
                            constrained_layout=True,
                            figsize=(10, 3))

    fig.suptitle(title, fontsize=12)


    # rewards
    cumulative_reward_2d = np.ma.mean(cumulative_reward_3d, axis=2)
    # cumulative_reward_1d = np.array(np.max(np.max(cumulative_reward_3d, axis=2),axis=0))
    plot_line_variance(axs[0], cumulative_reward_2d)
    plot_min_max(axs[0], cumulative_reward_2d)
    axs[0].set_title('Cumulative reward')
    axs[0].set_xlabel('Iteration #')


    # timesteps
    timesteps_2d = np.ma.mean(timesteps_3d, axis=2)
    # timesteps_1d = np.array(np.min(np.min(timesteps_3d, axis=2), axis=0))
    plot_line_variance(axs[1], timesteps_2d)
    plot_min_max(axs[1], timesteps_2d)
    axs[1].set_title('Timesteps per episode')
    axs[1].set_xlabel('Iteration #')

    plt.show()


def plot_line_variance(ax, data, delta=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.

    ax:     axis object where the plot will be drawn
    data:   data of shape (num_trials, timesteps)
    delta:  (optional) scaling of the standard deviation around the average
            if ommitted, delta = 1.'''

    avg = np.ma.average(data, axis=0)
    std = np.ma.std(data, axis=0)

    # ax.plot(avg + delta * std, 'r--', linewidth=0.5)
    # ax.plot(avg - delta * std, 'r--', linewidth=0.5)
    ax.fill_between(range(len(avg)),
                    avg + delta * std,
                    avg - delta * std,
                    facecolor='red',
                    alpha=0.2)
    ax.plot(avg)


# #############################################################################
#
# Experience
#
# #############################################################################

class Experience():
    def __init__(self, env, agent):
        pass


class agent():
    pass


def get_features(state, action):
    '''For a state [x, xdot] and action
    returns the feature vector as defined in Sutton, R. Reinforcement
    Learning, 2nd ed. (2018), Section 10.1, page 246''''

    x, xdot = state

    # create index hash table
    iht = tiling.IHT(args.size)
    features = tiling.tiles(
        iht, 8, [8 * x / (0.5 + 1.2), 8 * xdot / (0.07 + 0.07)], action
        )

    return features


def sarsa(env, lammbda, alpha, seed=None):

    assert alpha > 0
    assert 0 <= lammbda <= 1

    # initialize environement and weights
    env.seed(seed)
    env.reset()
    state = env.state()
    action = env.action_space.sample()
    features = get_features(state, action)
    w = np.zeros(features.shape)
    steps = 0

    for episode in range(args.episodes):
        state = env.state
        action = choose_action()
        z = np.zeros(features.shape)
        while not (done or steps > args.max_steps):
            steps += 1
            observation, reward, done, info = env.step(action)
            delta = reward
            features = get_features(observation, action)
            for i in :
                delta = delta - w[i]
                z[i] = 1



# #############################################################################
#
# Main
#
# #############################################################################


args = get_arguments()


def MountainCar():
    import gym

    env = gym.make('MountainCar-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()

def main():

    # sets the seed for random experiments
    np.random.seed(args.seed)

    # sets the environment
    MountainCar()


if __name__ == '__main__':
    main()
