#!/home/sunming/miniconda3/envs/normal/bin/python
# -*- coding: utf-8 -*-
#===================================================================================================
# File:           draw_contourf_plot_eof.py
# Category:       python script
# Author(s):      Hong Yutao 
# Date Created:   2026-01-12 by Hong Yutao
# Last Updated:   2026-01-12 by Hong Yutao
#---------------------------------------------------------------------------------------------------
# Function:       draw contourf and plot for EOF analysis
#===================================================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmaps
from matplotlib import rc
import os
import pandas as pd
from scipy.stats import linregress, ttest_1samp, ttest_ind
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature
import scipy.stats as stats
from pathlib import Path 
import scipy.stats as sts
from eofs.standard import Eof


def t_test_2p(arr1, arr2):
    t_statistic, p_value = stats.ttest_ind(arr1, arr2, axis=0, equal_var=False)
    return p_value

def get_auto_ticks(data_min, data_max, target_ticks=5):
    """
    更智能的刻度生成：
    - target_ticks: 期望大约出现的刻度数量
    """
    span = data_max - data_min
    if span == 0:
        return np.array([data_min])

    # 1. 定义气象/地理常用的“漂亮”步长候选集
    # 包含：0.5度, 1度, 2度, 5度, 10度, 15度, 20度, 30度...
    potential_steps = np.array([0.5, 1, 2, 2.5, 5, 10, 15, 20, 30, 45, 60])
    
    # 2. 计算理想步长并从候选集中找最接近的一个
    ideal_step = span / (target_ticks - 1)
    # 找到绝对差值最小的那个索引
    best_step_idx = np.abs(potential_steps - ideal_step).argmin()
    step = potential_steps[best_step_idx]
    
    # 3. 计算起始点和终止点
    # 确保刻度稍微超出或对齐数据边缘
    start = np.floor(data_min / step) * step
    end = np.ceil(data_max / step) * step
    
    ticks = np.arange(start, end + step, step)
    
    # 4. 极端情况微调：如果刻度太多了，就跳一个步长
    if len(ticks) > target_ticks + 3:
        step = potential_steps[min(best_step_idx + 1, len(potential_steps)-1)]
        ticks = np.arange(np.floor(data_min / step) * step, np.ceil(data_max / step) * step + step, step)
        
    return ticks

def get_symmetric_levels(data, n_levels=20, p=98):
    """
    自动生成对称且刻度规整的 levels
    """
    # 排除异常值影响
    limit = np.nanpercentile(np.abs(data), p)
    
    # 寻找一个“漂亮”的步长 (Power of 10)
    # 这步是为了让刻度看起来更专业
    magnitude = 10 ** np.floor(np.log10(limit))
    rounded_limit = np.ceil(limit / (magnitude / 2)) * (magnitude / 2)
    
    return np.linspace(-rounded_limit, rounded_limit, n_levels + 1)

def set_up_axes(ax, var, tick_range=None):
    # 如果没有手动指定 tick_range，则通过 var (xr.DataArray) 自动计算
    if tick_range is None:
        lon_min, lon_max = var.lon.min().item(), var.lon.max().item()
        lat_min, lat_max = var.lat.min().item(), var.lat.max().item()
        
        xticks = get_auto_ticks(lon_min, lon_max)
        yticks = get_auto_ticks(lat_min, lat_max)
    else:
        xticks, yticks = tick_range

    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
    
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.add_feature(cfeature.LAND, color='lightgrey', zorder=1)

def customize_font_sizes(ax, title_fontsize=16, label_fontsize=14, tick_fontsize=12, xlabelpad=10, ylabelpad=10):
    ax.title.set_fontsize(title_fontsize)
    ax.title.set_fontweight('bold')
    ax.xaxis.label.set_fontsize(label_fontsize)
    ax.yaxis.label.set_fontsize(label_fontsize)
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    ax.xaxis.labelpad = xlabelpad
    ax.yaxis.labelpad = ylabelpad
    ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)

def draw_contourf_single(ax, lon, lat, var, subtitle, self_range, cmap, ax_aspect):
    """
    使用 **kwargs 整合杂项参数
    可选参数：sig, self_range, tick_range, cmap, figsize, dpi 等
    """
    c1 = ax.contourf(lon, lat, var, transform=ccrs.PlateCarree(), cmap=cmap, levels=self_range, extend='both')
    ax.set_aspect(ax_aspect)
    plt.colorbar(c1, ax=ax, orientation='vertical', fraction=0.05)
    ax.set_title(subtitle, loc='left', pad=0.05, weight='semibold')
    return ax

def  draw_plot_single(ax, x, y, subtitle):
    p1 = ax.plot(x,y)
    ax.set_title(subtitle, loc='left', pad=0.05, weight='semibold')
    ax.set_ylim([-4, 4])
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_yticklabels((-4, -2, 0, 2, 4))
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.set_xlim(x[0], x[-1])
    return ax


def draw_contourf_plot(eofs, pcs, vars, pic_name, title, **kwargs):
    """
    使用 **kwargs 整合杂项参数
    可选参数：sig, self_range, tick_range, cmap, figsize, dpi 等
    """
    neof = eofs.shape[0]

    # 1. 提取参数并设置默认值
    sig = kwargs.get('sig', 0.05)
    self_range = kwargs.get('self_range', None)
    tick_range = kwargs.get('tick_range', None)
    cmap = kwargs.get('cmap', 'RdBu_r')
    figsize = kwargs.get('figsize', (9, 4))
    dpi = kwargs.get('dpi', 400)
    ax_aspect = kwargs.get('ax_aspect', 1.7)
    subtitle = kwargs.get('subtitle', 
                      [f"EOF{i+1}" for i in range(neof)] + [f"PC{i+1}" for i in range(neof)])

    # --- 绘图环境设置 ---
    rc('font', family='Times New Roman', weight='semibold')
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=figsize)

    # 用于存储 ax 的列表，方便后续调用
    ax_maps = [] # 存放带投影的 ax (左列)
    ax_ts   = [] # 存放普通 ax (右列)
    

    for i in range(neof):
        # --- 1. 左列：空间模式 (带地图投影) ---
        # subplot索引逻辑：行数=neof, 列数=2, 当前位置=2*i+1 (即 1, 3, 5...)
        central_lon = 180 if eofs.lon.max() > 180 else 0
        ax_m = fig.add_subplot(neof, 2, 2*i + 1, 
                            projection=ccrs.PlateCarree(central_longitude=central_lon))
        ax_maps.append(ax_m)
        
        # --- 2. 右列：时间系数 (普通坐标) ---
        # subplot索引逻辑：行数=neof, 列数=2, 当前位置=2*i+2 (即 2, 4, 6...)
        ax_t = fig.add_subplot(neof, 2, 2*i + 2)
        ax_ts.append(ax_t)
           
    # --- 核心绘图逻辑 ---
    # 确定 levels
    if self_range is not None:
        my_levels = self_range
    else:
        if np.isnan(eofs).all():
            my_levels = 20
        else:
            # 这里的 20 和 98 也可以写进 kwargs.get() 进一步灵活化
            my_levels = get_symmetric_levels(eofs, n_levels=20, p=98)


    for i in range(neof):
        # 1. 画左边
        # subtitle索引：前 neof 个是 EOF 标题
        draw_contourf_single(ax_maps[i], eofs.lon, eofs.lat, eofs[i], subtitle[i]+f"(Vars: {vars[i]:.2f})", my_levels, cmap, ax_aspect)  
        set_up_axes(ax_maps[i], eofs, tick_range)

        # 2. 画右边
        # subtitle索引：后 neof 个是 PC 标题
        draw_plot_single(ax_ts[i], pcs.time, pcs[:, i], subtitle[neof + i])
    
    for ax in ax_maps + ax_ts:
        for spine in ['top', 'right', 'bottom', 'left']:
            ax.spines[spine].set_linewidth(2)
            ax.grid(which="major", linestyle="-.", color="grey", alpha=0.6, zorder=4)
        customize_font_sizes(ax)
    for ax in ax_maps:
        ax.spines['geo'].set_linewidth(2)
    
    # 标题设置
    # fig.suptitle(title, fontsize=20, fontweight="bold")
    
    # --- 保存 ---
    plt.tight_layout()
    plt.savefig(pic_name, dpi=dpi)
    plt.close(fig)

def draw_contourf_plot_eof(arr, neof, pic_name, title, **kwargs):
    """
    绘制EOF和PC
    """
    # 记录是否为 xarray 对象
    is_xarray = isinstance(arr, xr.DataArray)
    
    solver = Eof(np.array(arr))
    eofs = solver.eofsAsCorrelation(neofs=neof)
    pcs = solver.pcs(npcs=neof, pcscaling=1)
    pcs_raw = solver.pcs(npcs=neof, pcscaling=0)
    vars = solver.varianceFraction(neigs=neof)
    
    for i in range(neof):
        pcs_std = np.std(pcs_raw[:, i])
        eofs[i, :] = eofs[i, :] * pcs_std

    if is_xarray:
        eofs = xr.DataArray(
            eofs.copy(), 
            coords={'num': np.arange(neof), 'lat': arr.lat, 'lon': arr.lon}, 
            dims=['num', 'lat', 'lon']
        )
        pcs = xr.DataArray(
            pcs.copy(), 
            coords={'time': arr.time, 'num': np.arange(neof)}, 
            dims=['time', 'num']
        )

    draw_contourf_plot(eofs, pcs, vars, pic_name, title, **kwargs)



if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(script_path)
    # 获取父目录的绝对路径
    parent_dir = os.path.dirname(script_dir)
    # 获取父目录的绝对路径
    parent_dir = os.path.dirname(parent_dir)