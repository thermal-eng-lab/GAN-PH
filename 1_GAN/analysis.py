# PLot Images
# Wasserstein GAN with gradient penalty
# date 2024.04.19(Fri)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.family'] ='Times New Roman'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 15 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['mathtext.fontset'] = 'cm'

class Images:
    def __init__(self):
        self.df = pd.read_table("./log.dat", sep="\s+")
        self.LIST = {"G_loss":2, "D_loss":4, "Wasser_d":6,"vf_loss":8, "ssa_loss":10,"gp_loss":12}

    def _plot_data(self,target,epoch):
        fig,ax = plt.subplots(figsize=(3,3),dpi=600)
        _num = self.LIST[target]
        loss = self.df.iloc[:,_num]

            # ----- raw data ----- #
        ax.plot(loss, linewidth=0.5, alpha=0.4)

        # ----- moving average ----- #
        _sample = 100
        _b = np.ones(_sample)/_sample
        loss_ave = np.convolve(loss, _b, mode="same")
        ax.plot(loss_ave, label=target)

        if target == "Wasser_d":
            ax.set_ylabel('Wasserstein Distance')
        else:
            ax.set_ylabel("Loss value")
            
        ax.set_xlabel('Iteration')
        ax.set_xlim(0,275*epoch+50)
        ax.grid(color='gray',linestyle=":")
        path = f"./{target}.png"
        fig.tight_layout()
        fig.savefig(path) 

    def plot_save(self):
        plot_list = ["G_loss", "D_loss", "gp_loss", "Wasser_d", "vf_loss", "ssa_loss"]
        for i in plot_list:
            self._plot_data(target=i, epoch=50)

        return 

