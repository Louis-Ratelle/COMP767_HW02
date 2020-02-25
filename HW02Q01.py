import matplotlib.pyplot as plt
import numpy as np
import argparse

import os
import sys

from time import time
from datetime import datetime

NOW = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SEED = 1
NB_EPISODES = 25
NB_RUNS = 50

seed_count = 4


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

    parser = argparse.ArgumentParser(description='Creating a randome walk between [0,1] with a step in [-0.2,0.2].')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Seed for the random number generator.')
    parser.add_argument('-episodes', '--nb_episodes', type=int, default=NB_EPISODES,
                        help='Number of episodes in each run. One run step is '
                             'Default: ' + str(NB_EPISODES))
    parser.add_argument('-runs', '--nb_runs', type=int, default=NB_RUNS,
                        help='Number of runs per combination. '
                             'Default: ' + str(NB_RUNS))

    return parser.parse_args()


# #############################################################################
#
# Plotting
#
# #############################################################################

def plot_all_variances(title, d_lammbda_to_alphas, d_msves_for_lambda):
    '''Creates the two required plots: cumulative_reward and number of timesteps
        per episode.'''

    fig, axs = plt.subplots(nrows=3, ncols=2,
                            constrained_layout=True,
                            sharey=True,
                            figsize=(11, 9))
    #plt.ylim(0.0, 1.0)

    #title = "hello"
    #fig.suptitle(title, fontsize=12)

    for id_ax, (lammbda, l_alphas) in enumerate(d_lammbda_to_alphas.items()):
        label = "lambda = {}".format(lammbda)
        color = "C" + str(id_ax)
        data  = d_msves_for_lambda[lammbda]
        plot_line_variance(axs, id_ax,lammbda, l_alphas, data, label, color, axis=0, delta=1)

    plt.show()

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




def plot_line_variance(axs, id_ax,lammbda, x_values, data, label, color, axis=0, delta=1):
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
    #fig, ax = plt.subplots(nrows=1, ncols=1,
                            #constrained_layout=True,
                            #sharey=True,
                            #figsize=(5,5))
    ax = axs[id_ax//len(axs[0]), id_ax % len(axs[0])]
    ax.fill_between(x_values,
                    avg + delta * std,
                    avg - delta * std,
                    facecolor=color,
                    alpha=0.2)
    ax.set_xlabel('$\\alpha$')
    ax.set_ylabel('msve')
    ax.set_title('msve (with std) for $\\lambda$={} as a function of $\\alpha$'.format(str(lammbda)))
    #ax.set_xlim([0, 1.0])
    ax.set_ylim([-0.2, 1.0])
    ax.legend()
    ax.plot(x_values, avg, label=label, color=color, marker='.')
    #plt.show()

"""
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
"""

def plot_lambdas_w_alphas(d_msve, title = None):
    for lammbda in d_msve.keys():
        d_lammbda = d_msve[lammbda]
        l_alphas_for_lammbda= []
        for alpha in d_lammbda.keys():
            l_alphas_for_lammbda.append(alpha)
        l_alphas_for_lammbda.sort()
        X = l_alphas_for_lammbda
        Y = [d_lammbda[alpha] for alpha in l_alphas_for_lammbda]
        plt.plot(X, Y, label = "$\\lambda$={}".format(lammbda))

    plt.ylim(0.0, 0.2)
    plt.xlabel('$\\alpha$')
    plt.ylabel('msve')
    plt.title('msve curves for different $\\lambda$\'s as a functions of $\\alpha$')
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
        return a

    def build_features(self, num):
        a = np.zeros(11*10)
        for pos in range(10):
            ind = int(np.ceil((num - self.shift_10_tilings[pos])/0.1))
            a[ind*10 + pos] = 1
        return a


class td_lambda_agent():
    def __init__(self,alpha, lammbda, tiling, args):
        self.alpha = alpha
        self.lammbda = lammbda
        self.args = args
        self.ws = np.array([np.zeros(11*10) for _ in range(self.args.nb_runs)])
        self.tiling = tiling
        self.average_w = np.zeros(11*10)
        self.msves = {}


    def train_all_runs(self):
        for run_id in range(len(self.ws)):
            global seed_count
            np.random.seed(seed_count)
            seed_count += 1
            self.train_one_run(run_id)
        self.average_w = np.mean(self.ws, axis=0)

    def train_one_run(self, run_id):
        for episode in range(self.args.nb_episodes):
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

    agents = {}

    d_lammbda_to_alphas = {}

    d_lammbda_to_alphas[0.0] = np.round(np.arange(13) * 0.01, 2)
    d_lammbda_to_alphas[0.2] = np.round(np.arange(14) * 0.01, 2)
    d_lammbda_to_alphas[0.4] = np.round(np.arange(14) * 0.01, 2)
    d_lammbda_to_alphas[0.6] = np.round(np.arange(12) * 0.01, 2)
    d_lammbda_to_alphas[0.75] = np.round(np.arange(10) * 0.01, 2)
    d_lammbda_to_alphas[0.95] = np.round(np.arange(5) * 0.01, 2)


    # d_lammbda_to_alphas[0.9] = np.round(np.arange(7) * 0.01, 2)
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
        print("Calculating for lambda={}".format(lammbda))
        d_msve[lammbda] = {}
        d_std_msve[lammbda] = {}
        nb_alphas = len(l_alphas)
        d_msves_for_lambda[lammbda] = np.zeros((args.nb_runs, nb_alphas))

        for alpha_pos, alpha in enumerate(l_alphas):
            #print("for lammbda = {} and alpha = {}".format(lammbda, alpha))
            agents[(alpha, lammbda)]= td_lambda_agent(alpha, lammbda, tilings, args)
            agents[(alpha, lammbda)].train_all_runs()

            valid_nums = np.round(np.arange(21) * 0.05,2)
            mean_square_error = agents[(alpha, lammbda)].msve(lammbda, valid_nums, alpha_pos, d_msves_for_lambda)
            #print("msve for lammbda = {} and alpha = {}: ".format(lammbda, alpha), mean_square_error)
            d_msve[lammbda][alpha] = mean_square_error
            std_of_mean_square_errors = agents[(alpha, lammbda)].variance_curves(valid_nums)
            #print("std  for lammbda = {} and alpha = {}: ".format(lammbda, alpha), std_of_mean_square_errors)
            d_std_msve[lammbda][alpha] = std_of_mean_square_errors

    plot_lambdas_w_alphas(d_msve)
    plot_all_variances("lambda = {}".format(lammbda), d_lammbda_to_alphas, d_msves_for_lambda)


if __name__ == '__main__':
    main()
