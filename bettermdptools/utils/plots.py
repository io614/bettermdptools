# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator


class Plots:
    @staticmethod
    def values_heat_map(data, title, size):
        data = np.around(np.array(data).reshape(size), 2)
        df = pd.DataFrame(data=data)
        sns.heatmap(df, annot=True).set_title(title)
        plt.show()

    @staticmethod
    def v_iters_plot(data, title):
        df = pd.DataFrame(data=data)
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=df, legend=None).set_title(title)
        plt.show()

    #modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def get_policy_map(pi, val_max, actions, map_size):
        """Map the best learned action to arrows."""
        #convert pi to numpy array
        best_action = np.zeros(val_max.shape[0], dtype=np.int32)
        for idx, val in enumerate(val_max):
            best_action[idx] = pi[idx]
        policy_map = np.empty(best_action.flatten().shape, dtype=str)
        for idx, val in enumerate(best_action.flatten()):
            policy_map[idx] = actions[val]
        policy_map = policy_map.reshape(map_size[0], map_size[1])
        val_max = val_max.reshape(map_size[0], map_size[1])
        return val_max, policy_map

    #modified from https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/
    @staticmethod
    def plot_policy(val_max, directions, map_size, title):
        """Plot the policy learned."""
        sns.heatmap(
            val_max,
            annot=directions,
            fmt="",
            cmap=sns.color_palette("Blues", as_cmap=True),
            linewidths=0.7,
            linecolor="black",
            xticklabels=[],
            yticklabels=[],
            annot_kws={"fontsize": "xx-large"},
        ).set(title=title)
        img_title = f"Policy_{map_size[0]}x{map_size[1]}.png"
        plt.show()
    
    @staticmethod
    def plot_policy_fl(V, pi, title, map_fl):
    ## Example map: ['SFFFF', 'FFHHF', 'FFFHF', 'FFHFH', 'HFFFG']
    ## Where S is start, H is hole, F is frozen, and G is goal

    ## Plot policy for FrozenLake
        map_size = int(math.sqrt(len(V)))
        V_square = V.reshape(map_size, map_size)
        # Get the policy map
        val_max, policy_map = Plots.get_policy_map(pi, V, ["←", "↓", "→", "↑"], (map_size, map_size))

        ## Plot the policy as heatmap of V values
        ## Color of tiles should represent V values. Plot a color bar on the side
        ## Borders of heatmap are black, except for start in blue, holes in red, and goal in green

        # Create a custom color map
        cmap = plt.get_cmap('RdYlGn')
        cmap = cmap(np.arange(cmap.N))
        cmap[:, -1] = 0.5  # Set alpha
        cmap = ListedColormap(cmap)

        # Plot the policy as heatmap of V values
        fig, ax = plt.subplots()
        im = ax.imshow(V_square, cmap=cmap)

        # Create borders for start, holes, and goal
        for i, row in enumerate(map_fl):
            for j, loc in enumerate(row):
                if loc == 'S':  # Start
                    ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='blue', lw=2))
                elif loc == 'H':  # Hole
                    ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', lw=2))
                elif loc == 'G':  # Goal
                    ax.add_patch(patches.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='green', lw=2))

        # Add the policy arrows
        for i in range(policy_map.shape[0]):
            for j in range(policy_map.shape[1]):
                ax.text(j, i, policy_map[i, j], ha='center', va='center', fontsize=15)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # Add a colorbar
        fig.colorbar(im, ax=ax)

        plt.title(title)
        # plt.show() 
            