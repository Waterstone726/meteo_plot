#!/home/sunming/miniconda3/envs/normal/bin/python

#===================================================================================================
# File:		draw_scatter_correlation.py
# Category:	python script
# Author(s):	Hong Yutao
# Date Created:	2024-01-10 by Hong Yutao
# Last Updated: 2024-01-10 by Hong Yutao
#---------------------------------------------------------------------------------------------------
# Function:	universal tempalte for scatter plot which contain correlation relationship
#===================================================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import cmaps
import time
from matplotlib import rc
import pandas as pd


def add_text(ax, str, loc="lower right", fontsize=12):
    ax.legend([str], loc=loc, handlelength=0, handletextpad=0, fontsize=fontsize)

def draw_trend(ax, x, y, get_slope=False):
    from scipy.stats import linregress
    x0 = np.array(x.copy())
    y0 = np.array(y.copy())
    x = x0[~np.isnan(x0)]
    y = y0[~np.isnan(x0)]
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    xlabel0 = np.sort(x)
    xlabel = np.linspace(xlabel0[0], xlabel0[-1], 100)
    y_reg = slope * xlabel + intercept
    ax.plot(xlabel, y_reg, color="#b91f24", zorder=0)
    # ax.text(0.6, 0.12, f"R = {r_value:.2f}\nSlope = {slope:.2e}", fontsize=12, color='k', transform=ax.transAxes, ha='left', va='top') #ha为水平方向对齐方式 va为垂直方向对齐方式
    add_text(ax, f"R = {r_value:.2f}\nSlope = {slope:.2e}")
    if get_slope:
        return slope, r_value, p_value

def customize_font_sizes(ax, title_fontsize=16, label_fontsize=14, tick_fontsize=12, xlabelpad=10, ylabelpad=10):
    """
    自定义 Matplotlib 图表的标题、x 和 y 轴标签、以及 x 和 y 轴刻度的字体大小。
    
    参数:
    ax (matplotlib.axes.Axes): 要自定义的图表的 Axes 对象。
    title_fontsize (int): 标题的字体大小。
    label_fontsize (int): x 和 y 轴标签的字体大小。
    tick_fontsize (int): x 和 y 轴刻度的字体大小。
    """
    # 设置标题的字体大小
    ax.title.set_fontsize(title_fontsize)
    ax.title.set_fontweight('bold')
    
    # 设置 x 轴和 y 轴标签的字体大小
    ax.xaxis.label.set_fontsize(label_fontsize)
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.labelpad = xlabelpad
    ax.yaxis.labelpad = ylabelpad
    
    # 设置 x 轴和 y 轴刻度的字体大小
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

# 次要刻度设置
def minor_tick(ax, num1, num2):
    from matplotlib.ticker import AutoMinorLocator
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=num1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=num2))



def draw_line(ax, alpha):
    ax.axhline(y=alpha, linestyle='--', linewidth=1, color='k', dashes=(5, 5))
    # ax.axhline(y=alpha/2, linestyle='-.', linewidth=1, color='k')
    # ax.axhline(y=0, linestyle=':', linewidth=1, color='k')
    # ax.axvline(x=0, linestyle=':', linewidth=1, color='k')


def draw_scatter_correlation(arr1, arr2, pic_name, label, title, order_text, lim, tick):
    plt.close()
    rc('font', family='Times New Roman', weight='semibold')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.titlepad'] = 30
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)  
        ax.spines[spine].set_linewidth(2)
    if order_text:
        for i in range(len(arr1)):
            ax.scatter(arr1[i], arr2[i], c="#66ece1", s=20)
            ax.text(arr1[i], arr2[i]+0.01, i+1, fontsize=9, color = "dimgray", weight="bold", style = "italic", verticalalignment='center', horizontalalignment='center')
    else:
        ax.scatter(arr1[i], arr2[i], c="#a7a7a7", s=20)

    draw_trend(ax, arr1, arr2)

    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    if tick is not None:
        ax.set_xticks(tick[0])
        ax.set_yticks(tick[1])
    
    minor_tick(ax, 2, 2)
    # ax.axhline(y=0, color='gray', linestyle='--')
    # ax.axvline(x=0, color='gray', linestyle='--')
    # ax.xaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
    # ax.yaxis.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
    ax.set_xlabel(label[0], fontweight='bold')
    ax.set_ylabel(label[1], fontweight='bold')
    # plt.tick_params(axis='both', which='major', length=6)
    # plt.tick_params(axis='both', which='minor', length=4)

    # fig.suptitle(title, fontsize=16, fontweight='bold')
    ax.set_title(title)

    customize_font_sizes(ax)
    plt.tight_layout()
    plt.savefig(pic_name, dpi=400)

if __name__ == '__main__':
    start = time.time()
    print('ᕕ( ᐛ )ᕗ')
    print('==========程序开始运行==========')


    print('==========程序运行结束==========')
    print('₍₍٩( ᐛ )۶₎₎♪')
    print('程序共运行了 %f 秒' % (time.time() - start))
