#!/home/sunming/miniconda3/envs/normal/bin/python
# -*- coding: utf-8 -*-
#===================================================================================================
# File:           draw_contourf_regression.py
# Category:       python script
# Author(s):      Hong Yutao 
# Date Created:   2026-01-05 by Hong Yutao
# Last Updated:   2026-01-05 by Hong Yutao
#---------------------------------------------------------------------------------------------------
# Function:       draw contourf for regression coefficient of 2d var pattern
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

def nanreg(x, y, edof=None):
    """
    NaN-safe linear regression with Pairwise Validity and EDOF support.
    y = slope * x + intercept

    Parameters
    ----------
    x : (N,) array-like
        Predictor
    y : (N, ...) array-like
        Predictand
    edof : float or array-like, optional
        Effective Degrees of Freedom. If None, uses N_valid - 2.
        If provided, should match the shape of the spatial grid or be a scalar.

    Returns
    -------
    slope, intercept, corr, p_value, N_valid_map
    """

    # 1. 维度处理
    x = np.asarray(x).ravel()
    y = np.asarray(y)
    N = x.shape[0]
    if y.shape[0] != N:
        raise ValueError("x.shape[0] must equal y.shape[0]")

    original_shape = y.shape[1:]
    y_rs = y.reshape(N, -1)
    P = y_rs.shape[1]

    # 2. 建立成对有效掩码 (Pairwise Mask)
    # 只有当 x 和 y 在同一时间点都有值时，才参与计算
    mask = np.isfinite(x[:, None]) & np.isfinite(y_rs)
    N_valid = mask.sum(axis=0).astype(float)
    
    # 预设无效位 (样本量太少或方差为0的格点)
    invalid_gate = (N_valid < 3)

    # 3. 高效计算各项累加和 (核心优化：避免产生巨大中间矩阵)
    # 将 NaN 置为 0 以便进行矩阵运算/求和，配合 mask 仅计算有效值
    x_filled = np.where(mask, x[:, None], 0.0)
    y_filled = np.where(mask, y_rs, 0.0)

    # 计算均值 (仅针对成对有效样本)
    s_x = x_filled.sum(axis=0)
    s_y = y_filled.sum(axis=0)
    m_x = s_x / N_valid
    m_y = s_y / N_valid

    # 计算离差平方和与协方差
    # 利用公式: Cov(X,Y) = E[XY] - E[X]E[Y]
    # 使用 einsum 或 dot 运算比直接乘法更省内存且更快
    ss_xx = (x_filled**2).sum(axis=0) - N_valid * (m_x**2)
    ss_yy = (y_filled**2).sum(axis=0) - N_valid * (m_y**2)
    ss_xy = (x_filled * y_filled).sum(axis=0) - N_valid * (m_x * m_y)

    # 4. 计算回归参数
    # 防止除以 0
    with np.errstate(divide='ignore', invalid='ignore'):
        slope = ss_xy / ss_xx
        intercept = m_y - slope * m_x
        
        # 相关系数并处理 R 值溢出
        corr = ss_xy / np.sqrt(ss_xx * ss_yy)
        corr = np.clip(corr, -1.0, 1.0)  # 解决溢出处理

    # 5. 显著性检验与有效自由度
    # 如果没提供 edof，则使用样本量-2
    if edof is None:
        df = N_valid - 2
    else:
        # 如果 edof 是标量或与 grid 一致的数组
        df = np.asarray(edof) - 2
    
    # 确保自由度大于 0
    df = np.where(df > 0, df, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        # t = r * sqrt(df / (1 - r^2))
        t_stat = corr * np.sqrt(df / (1.0 - corr**2 + 1e-20))
        p_value = 2 * (1 - sts.t.cdf(np.abs(t_stat), df))

    # 6. 掩码清理：将 N_valid < 3 或方差异常的格点设为 NaN
    slope[invalid_gate] = np.nan
    intercept[invalid_gate] = np.nan
    corr[invalid_gate] = np.nan
    p_value[invalid_gate] = np.nan

    # 7. Reshape 回原形状
    return (slope.reshape(original_shape), 
            intercept.reshape(original_shape), 
            corr.reshape(original_shape), 
            p_value.reshape(original_shape), 
            N_valid.reshape(original_shape))


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


def draw_contouf_regression(index, arr: xr.DataArray, pic_name, title, subtitle, **kwargs):
    """
    绘制回归系数
    """
    r_value, intercept, corr, p_value, N_valid = nanreg(index, arr)
    r_value = xr.DataArray(r_value, coords=[arr.lat, arr.lon], dims=["lat", "lon"])

    draw_contourf(r_value, p_value, pic_name, title, subtitle, **kwargs)



if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(script_path)
    # 获取父目录的绝对路径
    parent_dir = os.path.dirname(script_dir)
    # 获取父目录的绝对路径
    parent_dir = os.path.dirname(parent_dir)