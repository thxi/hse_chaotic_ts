import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def normalize(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Normalizes the array into a [-1, 1] range

    :param arr: the array to be normalized
    :return: tuple[normalized array, arr_min, arr_max]
    """
    ma, mi = arr.max(), arr.min()
    return (arr - arr.min()) / (arr.max() - arr.min()), mi, ma


def denormalize(arr: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Denormalizes the array

    :param arr: array to be denormalized
    :param min_val: arr_min
    :param max_val: arr_max
    :return: denormalized array
    """
    return arr * (max_val - min_val) + min_val


def gen_sin_wave(train_periods, test_periods, points_per_period):
    total_periods = (train_periods + test_periods)
    x = np.linspace(0, 2 * np.pi * total_periods,
                    total_periods * points_per_period)
    return np.sin(x)


def plot_trajectories(label,
                      X_train,
                      X_test,
                      noise_amp,
                      n_trajectories,
                      X_traj_pred,
                      X_pred,
                      filename=None):
    fig = plt.figure(figsize=[14, 7])

    train_size = X_train.shape[0]
    test_size = X_test.shape[0]

    fig.suptitle(
        f'{label}; train size={train_size}, test size={test_size}, n_trajectories={n_trajectories}, var={noise_amp}',
        fontsize=16)

    # series plot
    plt.subplot(2, 1, 1)
    plt.plot(X_train, label='train')
    plt.plot(range(train_size, train_size + test_size), X_test, label='test')

    plt.title(label)
    plt.vlines(train_size, 0, 1, color='orange', linestyle='dashed')
    plt.title('Train/test split')
    plt.legend(loc='upper right')

    # pred trajectories plot
    plt.subplot(2, 1, 2)
    plt.plot(X_test, label=label, zorder=1)

    # plt.ylim(-0.1, 1.1)

    for i in range(n_trajectories):
        plt.plot(X_traj_pred[:, i], c='orange', lw=0.5, zorder=0)

    plt.scatter(range(X_pred.size),
                X_pred,
                label='predicted',
                c='red',
                zorder=2)

    plt.title('Predicted trajectories (orange)')
    plt.legend(loc='upper right')

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)


def plot_runs(runs, h_max, filename=None):
    fig = plt.figure(figsize=(20, 10))

    # non-pred and rmse
    plt.subplot(1, 2, 1)

    colors = ['orange', 'blue', 'green', 'cyan', 'purple', 'red']
    for color, label in zip(colors, runs.keys()):
        plt.plot(range(1, h_max + 1),
                 runs[label]['non_pred'],
                 color=color,
                 label=label)

    plt.title(f"Non - Predictable Points")
    plt.xlim(1, h_max)
    plt.ylim(0, 100)
    plt.xlabel('Forecasting horizon')
    plt.ylabel('Percentage of non-predictable')
    plt.legend(loc='upper left')

    plt.subplot(1, 2, 2)

    for color, label in zip(colors, runs.keys()):
        plt.plot(range(1, h_max + 1),
                 runs[label]['rmse'],
                 color=color,
                 label=label)

    plt.title(f"RMSE")
    plt.xlim(1, h_max)
    plt.xlabel('Forecasting horizon')

    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)