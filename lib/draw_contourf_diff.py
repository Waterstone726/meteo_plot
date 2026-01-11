#!/home/sunming/miniconda3/envs/normal/bin/python
# -*- coding: utf-8 -*-
#===================================================================================================
# File:           draw_contourf_diff.py
# Category:       python script
# Author(s):      Hong Yutao 
# Date Created:   2026-01-08 by Hong Yutao
# Last Updated:   2026-01-08 by Hong Yutao
#---------------------------------------------------------------------------------------------------
# Function:       draw contourf for difference of 2 variables with Welch's t-test
#===================================================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmaps
from matplotlib import rc
import os
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature
import scipy.stats as stats
from pathlib import Path 

def calculate_diff_stats(arr1, arr2):
    """
    Calculate the difference between means of two arrays and perform Welch's t-test.
    
    Parameters
    ----------
    arr1, arr2 : xarray.DataArray
        Input arrays with shape (Time/Member, Lat, Lon).
        
    Returns
    -------
    diff_mean : xarray.DataArray
        Difference of means (Mean1 - Mean2).
    p_value : xarray.DataArray
        P-value from Welch's t-test.
    """
    # 1. 计算时间/样本维度的平均值差 (假设第0维是样本维)
    # xarray 的 mean 会自动处理维度名称，这里假设名为 'time' 或 'member'，
    # 或者直接使用 axis=0 更加通用
    mean1 = arr1.mean(dim=arr1.dims[0], skipna=True)
    mean2 = arr2.mean(dim=arr2.dims[0], skipna=True)
    diff_mean = mean1 - mean2
    
    # 2. Welch's t-test (双样本异方差t检验)
    # equal_var=False 即执行 Welch's t-test
    # nan_policy='omit' 自动忽略 NaN 值
    t_stat, p_val = stats.ttest_ind(arr1.values, arr2.values, axis=0, equal_var=False, nan_policy='omit')
    
    # 3. 将 numpy 结果包装回 xarray，保持经纬度坐标以便绘图
    p_value = xr.DataArray(p_val, coords=mean1.coords, dims=mean1.dims)
    
    return diff_mean, p_value

def get_auto_ticks(data_min, data_max, target_ticks=5):
    """
    更智能的刻度生成：
    - target_ticks: 期望大约出现的刻度数量
    """
    span = data_max - data_min
    if span == 0:
        return np.array([data_min])

    # 1. 定义气象/地理常用的“漂亮”步长候选集
    potential_steps = np.array([0.5, 1, 2, 2.5, 5, 10, 15, 20, 30, 45, 60])
    
    # 2. 计算理想步长并从候选集中找最接近的一个
    ideal_step = span / (target_ticks - 1)
    best_step_idx = np.abs(potential_steps - ideal_step).argmin()
    step = potential_steps[best_step_idx]
    
    # 3. 计算起始点和终止点
    start = np.floor(data_min / step) * step
    end = np.ceil(data_max / step) * step
    
    ticks = np.arange(start, end + step, step)
    
    # 4. 极端情况微调
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

def draw_contourf(var, p_value, pic_name, title, subtitle, **kwargs):
    """
    使用 **kwargs 整合杂项参数
    可选参数：sig, self_range, tick_range, cmap, figsize, dpi 等
    """
    # 1. 提取参数并设置默认值
    sig = kwargs.get('sig', 0.05)
    self_range = kwargs.get('self_range', None)
    tick_range = kwargs.get('tick_range', None)
    cmap = kwargs.get('cmap', 'RdBu_r')
    figsize = kwargs.get('figsize', (8, 5))
    dpi = kwargs.get('dpi', 400)

    # --- 绘图环境设置 ---
    rc('font', family='Times New Roman', weight='semibold')
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=figsize)
    
    # 动态计算中央经度
    central_lon = 180 if var.lon.max() > 180 else 0
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=central_lon))
    
    # --- 核心绘图逻辑 ---
    # 确定 levels
    if self_range is not None:
        my_levels = self_range
    else:
        if np.isnan(var).all():
            my_levels = 20
        else:
            # 这里的 20 和 98 也可以写进 kwargs.get() 进一步灵活化
            my_levels = get_symmetric_levels(var, n_levels=20, p=98)
    
    c1 = ax.contourf(var.lon, var.lat, var, transform=ccrs.PlateCarree(), 
                     cmap=cmap, levels=my_levels, extend='both')
    
    # 显著性打点 (使用提取出的 sig)
    ax.contourf(var.lon, var.lat, p_value, transform=ccrs.PlateCarree(), 
                levels=[0, sig, 1], hatches=['.', None], colors="none", zorder=2)
    
    # --- 细节打磨 ---
    ax.set_aspect(1.7)
    plt.colorbar(c1, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
    
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_linewidth(2)
    
    ax.grid(which="major", linestyle="-.", color="grey", alpha=0.6, zorder=4)
    
    # 调用外部配置函数
    set_up_axes(ax, var, tick_range)
    
    # 标题设置
    ax.set_title(subtitle, fontsize=16, fontweight="semibold", loc='left', pad=6)
    customize_font_sizes(ax)
    fig.suptitle(title, fontsize=20, fontweight="bold")
    
    # --- 保存 ---
    plt.tight_layout()
    plt.savefig(pic_name, dpi=dpi)
    plt.close(fig)

def draw_contourf_diff(arr1, arr2, pic_name, title, subtitle, **kwargs):
    """
    Main function to calculate difference and plot.
    
    Parameters
    ----------
    arr1, arr2 : xarray.DataArray
        Input arrays (Time, Lat, Lon).
    pic_name : str
        Output file path.
    title, subtitle : str
        Plot titles.
    tick_range : list or None
        [xticks, yticks].
    self_range : list or None
        Specific levels for contourf.
    """
    # 计算均值差和 Welch's t-test 显著性
    diff_val, p_value = calculate_diff_stats(arr1, arr2)
    
    # 绘图
    draw_contourf(diff_val, p_value, pic_name, title, subtitle, **kwargs)

if __name__ == "__main__":
    # 示例用法
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    print(f"Script initialized at: {script_path}")
    
    # Mock data example (uncomment to test)
    # lat = np.linspace(-30, 30, 60)
    # lon = np.linspace(120, 240, 120)
    # arr1 = xr.DataArray(np.random.randn(100, 60, 120), coords=[range(100), lat, lon], dims=['time', 'lat', 'lon'])
    # arr2 = xr.DataArray(np.random.randn(100, 60, 120) + 0.5, coords=[range(100), lat, lon], dims=['time', 'lat', 'lon'])
    # draw_contourf_diff(arr1, arr2, "test_diff.png", "Test Diff", "Difference (A - B)", tick_range=None)