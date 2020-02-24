import matplotlib.pyplot as plt
import numpy as np
import argparse

import os
import sys

from time import time
from datetime import datetime

# constants
ARMS = 10
RUNS = 10
STEPS_PER_RUN = 1000

NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SEED = None
# SEED = 197710

NB_EPISODES = 25 # 25
NB_RUNS = 50 # 50

seed_count = 4

#set_LFPR = set()

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

    parser = argparse.ArgumentParser(description='Creating a k-armed bandit.')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('-k', '--arms', type=int, default=ARMS,
                        help='Number of arms on the bandit. Default: '
                        + str(ARMS))
    parser.add_argument('-n', '--runs', type=int, default=RUNS,
                        help='Number of runs to be executed. Default: '
                        + str(RUNS))
    parser.add_argument('-s', '--steps', type=int, default=STEPS_PER_RUN,
                        help='Number of steps in each run. One run step is '
                        'the ensemble of training steps and testing steps. '
                        'Default: ' + str(STEPS_PER_RUN))

    return parser.parse_args()


# #############################################################################
#
# Plotting
#
# #############################################################################

"""
def plot_line_variance(ax, data, gamma=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.

    ax:     axis object where the plot will be drawn
    data:   data of shape (num_trials, timesteps)
    gamma:  (optional) scaling of the standard deviation around the average
            if ommitted, gamma = 1.'''

    avg = np.average(data, axis=0)
    std = np.std(data, axis=0)

    # ax.plot(avg + gamma * std, 'r--', linewidth=0.5)
    # ax.plot(avg - gamma * std, 'r--', linewidth=0.5)
    ax.fill_between(range(len(avg)),
                    avg + gamma * std,
                    avg - gamma * std,
                    facecolor='red',
                    alpha=0.2)
    ax.plot(avg)
"""

def plot_all_variances(title, d_lammbda_to_alphas, d_msves_for_lambda):
    '''Creates the two required plots: cumulative_reward and number of timesteps
        per episode.'''

    #fig, axs = plt.subplots(nrows=1, ncols=6,
                            #constrained_layout=True,
                            #sharey=True,
                            #figsize=(10, 3))

    #fig.suptitle(title, fontsize=12)

    for id_ax, (lammbda, l_alphas) in enumerate(d_lammbda_to_alphas.items()):
        label = "lambda = {}".format(lammbda)
        color = "C" + str(id_ax)
        data  = d_msves_for_lambda[lammbda]
        plot_line_variance(l_alphas, data, label, color, axis=0, delta=1)

    #plt.show()

    """
    plot_lambdas(axs[0], steps, x_values=alphas, series=lambdas)
    axs[0].set_xlabel('alpha')
    axs[0].set_ylabel('Average steps')
    axs[0].set_title('Sarsa($\\lambda$)')
    axs[0].legend()
    # axs[0].set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']))

    plot_alphas(axs[1], steps, x_values=lambdas, series=alphas)
    axs[1].set_xlabel('lambda')
    axs[1].set_ylabel('Average steps')
    axs[1].set_title('Learning rate')
    axs[1].legend()

    plot_alphas2(axs[1], steps, x_values=lambdas, series=alphas)
    axs[1].set_xlabel('lambda')
    axs[1].set_ylabel('Average steps')
    axs[1].set_title('Learning rate')
    axs[1].legend()
    """




def plot_line_variance(x_values, data, label, color, axis=0, delta=1):
    '''Plots the average data for each time step and draws a cloud
    of the standard deviation around the average.
    Input:
    ax      : axis object where the plot will be drawn
    data    : data of shape (num_trials, timesteps)
    color   : the color to be used
    delta   : (optional) scaling of the standard deviation around the average
              if ommitted, delta = 1.'''

    avg = np.average(data, axis)
    std = np.std(data, axis)

    # ax.plot(avg + delta * std, color + '--', linewidth=0.5)
    # ax.plot(avg - delta * std, color + '--', linewidth=0.5)
    fig, ax = plt.subplots(nrows=1, ncols=1,
                            constrained_layout=True,
                            sharey=True,
                            figsize=(5,5))
    ax.fill_between(x_values,
                    avg + delta * std,
                    avg - delta * std,
                    facecolor=color,
                    alpha=0.2)
    ax.plot(x_values, avg, label=label, color=color, marker='.')
    plt.show()


def plot4(title, training_return, training_regret, testing_reward, testing_regret):
    '''Creates the four required plots: average training return, training regret,
    testing policy reward and testing regret.'''

    fig, axs = plt.subplots(nrows=2, ncols=2,
                            constrained_layout=True,
                            figsize=(10, 6))

    fig.suptitle(title, fontsize=12)

    plot_line_variance(axs[0, 0], training_return)
    axs[0, 0].set_title('Training return')

    plot_line_variance(axs[0, 1], training_regret)
    axs[0, 1].set_title('Total training regret')

    plot_line_variance(axs[1, 0], testing_reward)
    axs[1, 0].set_title('Policy reward')
    axs[1, 0].set_ylim(bottom=0)

    plot_line_variance(axs[1, 1], testing_regret)
    axs[1, 1].set_title('Total testing regret')


def plot_lambdas_w_alphas(d_msve, title = None):
    for lammbda in d_msve.keys():
        d_lammbda = d_msve[lammbda]
        l_alphas_for_lammbda= []
        for alpha in d_lammbda.keys():
            l_alphas_for_lammbda.append(alpha)
        l_alphas_for_lammbda.sort()
        X = l_alphas_for_lammbda
        Y = [d_lammbda[alpha] for alpha in l_alphas_for_lammbda]
        #print("x: ", X)
        #print("y: ", Y)
        plt.plot(X, Y, label = lammbda)

    plt.ylim(0.0, 0.2)
    plt.legend()
    plt.show()


# #############################################################################
#
# Helper functions
#
# #############################################################################


def softmax(x):
    '''Softmax implementation for a vector x.'''

    # subtract max for numerical stability
    # (does not change result because of identity softmax(x) = softmax(x + c))
    z = x - max(x)

    return np.exp(z) / np.sum(np.exp(z), axis=0)


def random_argmax(vector):
    '''Select argmax at random... not just first one.'''

    index = np.random.choice(np.where(vector == vector.max())[0])

    return index


# #############################################################################
#
# This will create the tile coding
#
# #############################################################################


class tiling10():
    def __init__(self, seed = 3):
        np.random.seed(seed)
        self.shift_10_tilings = self.create_tilings()


    def create_tilings(self):
        a = np.random.uniform(low=0.0, high=.1,size=10)
        a = np.sort(a)
        #print("tiling: ", a)
        return a
        #return np.random.uniform(low=0.0, high=.1,size=10)

    def build_features(self, num):
        a = np.zeros(11*10)
        for pos in range(10):
            ind = int(np.ceil((num - self.shift_10_tilings[pos])/0.1))
            #a[ind + pos * 11] = 1
            #set_LFPR.add(ind + pos * 11)
            a[ind*10 + pos] = 1
            #set_LFPR.add(ind*10 + pos)
        return a


class td_lambda_agent():
    def __init__(self,alpha, lammbda, tiling):
        self.alpha = alpha
        self.lammbda = lammbda
        self.ws = np.array([np.zeros(11*10) for _ in range(NB_RUNS)]) # is there a bias here, I assume no ??
        # In addition, add a feature corresponding to each interval that takes the value 1 when the state was within that tile, and 0 otherwise ??
        self.tiling = tiling
        self.average_w = np.zeros(11*10)
        self.msves = {}

    def train_all_runs(self):
        for run_id in range(len(self.ws)):
            global seed_count
            np.random.seed(seed_count)
            #print("seed_count: ", seed_count)
            seed_count += 1
            self.train_one_run(run_id)
        self.average_w = np.mean(self.ws, axis=0)
        #return self.ws

    def train_one_run(self, run_id):
        for episode in range(NB_EPISODES):
            self.ws[run_id] = self.train_one_episode(self.ws[run_id])

    def train_one_episode(self, w):
        z = np.zeros(110)
        s = 0.5
        cont = True
        while cont:
            f_s = self.tiling.build_features(s)
            sp = s + np.random.uniform(low=-0.2, high=0.2)
            if sp >1.0 or sp < 0.0:
                r = sp
                cont = False
            else:
                r = 0
                f_sp = self.tiling.build_features(sp)
            z = self.lammbda * z + f_s
            v_s = np.sum(w * f_s)
            if cont:
                v_sp = np.sum(w * f_sp)
            else:
                v_sp = 0
            delta = r + v_sp - v_s
            w = w + self.alpha * delta * z
            s = sp
        return w

    def msve(self, lammbda, l_nums, alpha_pos, d_msves_for_lambda):
        assert len(l_nums) > 0
        l_msve = []

        for run_id, w in enumerate(self.ws):
            tot = 0
            for num in l_nums:
                v = np.sum(w * self.tiling.build_features(num))
                #print("v={} for num={}".format(v, num))
                tot += (v - num)**2
            l_msve.append(tot / len(l_nums))
            d_msves_for_lambda[lammbda][run_id, alpha_pos] = tot / len(l_nums)
        self.l_msve = l_msve
        return np.mean(self.l_msve)

    def variance_curves(self, l_nums):
        return np.std(self.l_msve)
        """
        assert len(l_nums) > 0
        l_variances = []
        for w in self.ws:
            tot = 0
            for num in l_nums:
                v = np.sum(w * self.tiling.build_features(num))
                # print("v={} for num={}".format(v, num))
                tot += (v - num) ** 2
            l_variances.append(tot / len(l_nums))
        #print(l_variances)
        std = np.std(l_variances)
        return std
        """





# #############################################################################
#
# Main
#
# #############################################################################


def main():

    # parses command line arguments
    args = get_arguments()

    tilings= tiling10(5)
    #print("tiling:", tilings.shift_10_tilings)
    """
    for i in range(21):
        print(i*0.05, tilings.build_features(i*0.05))
    print("0.32:", tilings.build_features(0.32))
    print("0.33:", tilings.build_features(0.33))
    """

    agents = {}


    d_lammbda_to_alphas = {}

    d_lammbda_to_alphas[0.0] = np.round(np.arange(13) * 0.01, 2)
    d_lammbda_to_alphas[0.2] = np.round(np.arange(14) * 0.01, 2)
    d_lammbda_to_alphas[0.4] = np.round(np.arange(14) * 0.01, 2)
    d_lammbda_to_alphas[0.6] = np.round(np.arange(12) * 0.01, 2)
    d_lammbda_to_alphas[0.75] = np.round(np.arange(10) * 0.01, 2)
    #d_lammbda_to_alphas[0.9] = np.round(np.arange(7) * 0.01, 2)
    d_lammbda_to_alphas[0.95] = np.round(np.arange(5) * 0.01, 2)
    #d_lammbda_to_alphas[0.98] = np.round(np.arange(6) * 0.01, 2)
    #d_lammbda_to_alphas[1.0] = np.round(np.arange(3) * 0.01, 2)


    """
    # d_lammbda_to_alphas[0.1] = np.round(np.arange(15) * 0.01, 2)
    # d_lammbda_to_alphas[0.2] = np.round(np.arange(16) * 0.01, 2)
    # d_lammbda_to_alphas[0.3] = np.round(np.arange(15) * 0.01, 2)
    # d_lammbda_to_alphas[0.5] = np.round(np.arange(15) * 0.01, 2)
    # d_lammbda_to_alphas[0.6] = np.round(np.arange(14) * 0.01, 2)
    # d_lammbda_to_alphas[0.7] = np.round(np.arange(13) * 0.01, 2)
    # d_lammbda_to_alphas[0.8] = np.round(np.arange(11) * 0.01, 2)
    #d_lammbda_to_alphas[0.3] = np.round([0.25], 2)
    """


    d_msve = {}
    d_std_msve = {}
    d_msves_for_lambda = {}

    for lammbda, l_alphas in d_lammbda_to_alphas.items():
        d_msve[lammbda] = {}
        d_std_msve[lammbda] = {}
        nb_alphas = len(l_alphas)
        d_msves_for_lambda[lammbda] = np.zeros((NB_RUNS, nb_alphas))

        for alpha_pos, alpha in enumerate(l_alphas):
            print("for lammbda = {} and alpha = {}".format(lammbda, alpha))
            #alpha = 0.1
            #lammbda = 0.5
            agents[(alpha, lammbda)]= td_lambda_agent(alpha, lammbda, tilings)
            agents[(alpha, lammbda)].train_all_runs()
            #print("ws: ", agents[(alpha, lammbda)].ws)
            #print("average w: :", agents[(alpha, lammbda)].average_w)

            valid_nums = np.round(np.arange(21) * 0.05,2)
            mean_square_error = agents[(alpha, lammbda)].msve(lammbda, valid_nums, alpha_pos, d_msves_for_lambda)
            print("msve for lammbda = {} and alpha = {}: ".format(lammbda, alpha), mean_square_error)
            d_msve[lammbda][alpha] = mean_square_error
            std_of_mean_square_errors = agents[(alpha, lammbda)].variance_curves(valid_nums)
            print("std  for lammbda = {} and alpha = {}: ".format(lammbda, alpha), std_of_mean_square_errors)
            d_std_msve[lammbda][alpha] = std_of_mean_square_errors


    #print("set: ", set_LFPR)
    #print(len(set_LFPR))

    #print(d_msve)

    plot_lambdas_w_alphas(d_msve)
    #print(d_msves_for_lambda)
    plot_all_variances("lambda = {}".format(lammbda), d_lammbda_to_alphas, d_msves_for_lambda)


if __name__ == '__main__':
    main()
