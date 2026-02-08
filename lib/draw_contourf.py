#!/home/sunming/miniconda3/envs/normal/bin/python
# -*- coding: utf-8 -*-
#===================================================================================================
# File:           draw_contourf_simple.py
# Function:       Draw 2D spatial pattern without significance test (No p_value needed)
#===================================================================================================

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import rc
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from matplotlib.ticker import AutoMinorLocator
import cartopy.feature as cfeature
import os

# 尝试导入 cmaps，如果用户没有安装该库，则不会报错，但需要手动指定标准 cmap
try:
    import cmaps
except ImportError:
    pass

# =============================================================================
# Helper Functions (保留你原有的辅助函数)
# =============================================================================

def get_auto_ticks(data_min, data_max, target_ticks=5):
    """
    更智能的刻度生成
    """
    span = data_max - data_min
    if span == 0:
        return np.array([data_min])

    potential_steps = np.array([0.5, 1, 2, 2.5, 5, 10, 15, 20, 30, 45, 60])
    ideal_step = span / (target_ticks - 1)
    best_step_idx = np.abs(potential_steps - ideal_step).argmin()
    step = potential_steps[best_step_idx]
    
    start = np.floor(data_min / step) * step
    end = np.ceil(data_max / step) * step
    
    ticks = np.arange(start, end + step, step)
    
    if len(ticks) > target_ticks + 3:
        step = potential_steps[min(best_step_idx + 1, len(potential_steps)-1)]
        ticks = np.arange(np.floor(data_min / step) * step, np.ceil(data_max / step) * step + step, step)
        
    return ticks

def get_symmetric_levels(data, n_levels=20, p=98):
    """
    自动生成对称且刻度规整的 levels (适用于异常值/系数)
    """
    limit = np.nanpercentile(np.abs(data), p)
    
    # 防止全NaN或全0导致的错误
    if np.isnan(limit) or limit == 0:
        return np.linspace(-1, 1, n_levels + 1)

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

# =============================================================================
# Main Function: Modified draw_contourf
# =============================================================================

def draw_contourf(var, pic_name, title, subtitle, **kwargs):
    """
    绘制2D空间场 (无需 p_value)
    
    Parameters
    ----------
    var : xarray.DataArray
        包含 lat 和 lon 维度的二维数组
    pic_name : str
        保存图片的文件名
    title : str
        总标题
    subtitle : str
        子标题
    **kwargs : 
        self_range : list or array, 自定义 levels
        tick_range : tuple, (xticks, yticks)
        cmap : str or colormap object
        figsize : tuple
        dpi : int
    """
    # 1. 提取参数并设置默认值
    self_range = kwargs.get('self_range', None)
    tick_range = kwargs.get('tick_range', None)
    # 默认 cmap 使用 RdBu_r，你可以根据需要改为 'WhiteBlueGreenYellowRed' 等 cmaps 库颜色
    cmap = kwargs.get('cmap', 'RdBu_r') 
    figsize = kwargs.get('figsize', (8, 5))
    dpi = kwargs.get('dpi', 400)

    # --- 绘图环境设置 ---
    rc('font', family='Times New Roman', weight='semibold')
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=figsize)
    
    # 动态计算中央经度 (太平洋视角 vs 0度视角)
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
            # 使用原来的对称逻辑。如果是全正值（如降水），建议在调用时通过 self_range 指定
            my_levels = get_symmetric_levels(var, n_levels=20, p=98)
    
    # 绘图
    c1 = ax.contourf(var.lon, var.lat, var, transform=ccrs.PlateCarree(), 
                     cmap=cmap, levels=my_levels, extend='both', zorder=0)
    
    # --- 细节打磨 ---
    ax.set_aspect(1.7) # 保持原比例
    
    # Colorbar
    plt.colorbar(c1, ax=ax, orientation='horizontal', fraction=0.05, pad=0.1)
    
    # 边框加粗
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_linewidth(2)
    
    # 网格线
    ax.grid(which="major", linestyle="-.", color="grey", alpha=0.6, zorder=4)
    
    # 调用外部配置函数 (设置刻度、地图要素)
    set_up_axes(ax, var, tick_range)
    
    # 标题设置
    ax.set_title(subtitle, fontsize=16, fontweight="semibold", loc='left', pad=6)
    customize_font_sizes(ax)
    fig.suptitle(title, fontsize=20, fontweight="bold")
    
    # --- 保存 ---
    plt.tight_layout()
    # 检查目录是否存在，不存在则创建
    output_dir = os.path.dirname(pic_name)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.savefig(pic_name, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {pic_name}")

# =============================================================================
# Usage Example
# =============================================================================
if __name__ == "__main__":
    # 构造一个假的测试数据
    lon = np.linspace(60, 150, 91)
    lat = np.linspace(-10, 60, 71)
    data = np.random.randn(len(lat), len(lon))
    
    # 必须构建成 xarray.DataArray，因为函数内部需要 .lon 和 .lat
    var_da = xr.DataArray(data, coords=[lat, lon], dims=['lat', 'lon'], name='test_data')
    
    # 调用函数
    draw_contourf(
        var=var_da,
        pic_name='./test_figure.png',
        title='Test Figure',
        subtitle='Random Data Pattern',
        cmap='RdBu_r', # 或者 cmaps.BlueWhiteOrangeRed
        # self_range=np.linspace(-3, 3, 21) # 可选：手动指定范围
    )
