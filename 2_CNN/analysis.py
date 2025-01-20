# analize the result of the training
# Surrogate model for estimating the specific surface area

# -------- import library -------- #
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- settings graph -------- #
plt.rcParams['font.family'] ='Times New Roman'#使用するフォント
plt.rcParams['xtick.direction'] = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction'] = 'in'#y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['xtick.major.width'] = 1.0#x軸主目盛り線の線幅
plt.rcParams['ytick.major.width'] = 1.0#y軸主目盛り線の線幅
plt.rcParams['font.size'] = 10 #フォントの大きさ
plt.rcParams['axes.linewidth'] = 1.0# 軸の線幅edge linewidth。囲みの太さ
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['mathtext.fontset'] = 'cm'
# centimeters in inches
cm = 1/2.54

# -------- Functions -------- #
def plot_moving_ave(data):
    train_loss = data.iloc[:,3]
    valid_loss = data.iloc[:,6]

    fig, ax = plt.subplots(figsize=(5,5),dpi=600)

    # -------- Plot original loss value -------- #
    ax.plot(train_loss, linewidth=0.5, alpha=0.4, color="midnightblue")
    ax.plot(valid_loss, linewidth=0.5, alpha=0.4, color="maroon")

    # -------- Plot moving average of loss value -------- #
    num = 5 
    b = np.ones(num) / num

    train_loss_ave = np.convolve(train_loss, b, mode = "same")
    valid_loss_ave = np.convolve(valid_loss, b, mode = "same")
    ax.plot(train_loss_ave, label='Train_loss',linewidth=1.0, color="midnightblue")
    ax.plot(valid_loss_ave, label='Valid_loss',linewidth=1.0, color="maroon")

    # -------- Setting of plot Image -------- #
    ax.legend(fancybox=False,edgecolor='black',framealpha=0.9)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss value')
    # ax.set_yscale("log")
    ax.grid(color='gray',linestyle=":")
    path = "./moving_ave.png"
    fig.savefig(path)

    return