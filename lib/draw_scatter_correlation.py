#!/home/sunming/miniconda3/envs/normal/bin/python

#===================================================================================================
# File:		quadratic_reg.py
# Category:	python script
# Author(s):	Hong Yutao
# Date Created:	2024-01-10 by Hong Yutao
# Last Updated: 2024-01-10 by Hong Yutao
#---------------------------------------------------------------------------------------------------
# Function:	cal quadratic regression of pc1 & pc2
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
import sys
sys.path.append("/home/hongyt/code/python_scripts")
from cmp import run_time, minor_tick
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
    ax.plot(xlabel, y_reg, color='r')
    # ax.text(0.6, 0.12, f"R = {r_value:.2f}\nSlope = {slope:.2e}", fontsize=12, color='k', transform=ax.transAxes, ha='left', va='top') #ha为水平方向对齐方式 va为垂直方向对齐方式
    add_text(ax, f"R = {r_value:.2f}\nSlope = {slope:.2e}")
    if get_slope:
        return slope, r_value, p_value

def quadratic_reg(x, y, ax, line_color, text_name):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    x = np.array(x)
    y = np.array(y)
    # 将 x 转换为二次多项式特征
    poly_features = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))

    # 使用线性回归模型拟合数据
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)

    # 打印系数 a, b, 和 c
    a, b, c = lin_reg.coef_[1], lin_reg.coef_[0], lin_reg.intercept_
    func = text_name + f"{a:.2f}" + r"$x^{2}$+" + f"{b:.2f}" + "x+" + f"{c:.2f}"

    # 绘制拟合曲线
    x_new = np.linspace(-2.5, 3.5, 200).reshape(-1, 1)
    X_new_poly = poly_features.transform(x_new)
    y_new = lin_reg.predict(X_new_poly)
    ax.plot(x_new, y_new, c=line_color, linewidth=1, label=func)
    return a

  
def sel_DJF(arr):
    arr_DJF = arr[:, arr.time.dt.month.isin([12, 1, 2])]
    return arr_DJF

def DJF_mean(arr):
    arr.coords['winter_year'] = ('time', arr['time.year'].data + (arr['time.month'].data == 12).astype(int))
    winter_arr = arr.sel(time=arr['time.month'].isin([12, 1, 2]))
    winter_mean = winter_arr.groupby('winter_year').mean()
    winter_year = winter_mean.winter_year
    time = pd.to_datetime([f"{year-1}-12-01" for year in winter_year.data], format="%Y-%m-%d")
    winter_mean = winter_mean.assign_coords(winter_year=time)
    return winter_mean[:, 1:]

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

def cal_quadratic_reg(pc1, pc2, pc1_DJF, pc2_DJF, pic_name, label):
    plt.close()
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    alpha_DJF = quadratic_reg(pc1_DJF, pc2_DJF, ax, 'r', text_name="DJF: ")
    alpha = quadratic_reg(pc1, pc2, ax, 'lightcoral', text_name="monthly: ")

    return alpha, alpha_DJF


@run_time
def draw_scatter(e_index, c_index, e_index_DJF, c_index_DJF, pic_name, label):
    plt.close()
    rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    # ax.scatter(e_index, c_index, c='lightgray', s=5)
    num_all_DJF = len(e_index_DJF)
    # EPE1_2, EPL
    num_EPE1 = len(e_index_DJF[(e_index_DJF>1) & (e_index_DJF<2) & (c_index_DJF>-1) & (c_index_DJF<1)])
    num_EPE2 = len(e_index_DJF[(e_index_DJF>2) & (c_index_DJF>-1) & (c_index_DJF<1)])
    num_EPE = len(e_index_DJF[(e_index_DJF>1) & (c_index_DJF>-1) & (c_index_DJF<1)])
    num_EPL = len(e_index_DJF[(e_index_DJF<-1) & (c_index_DJF>-1) & (c_index_DJF<1)])
    # if num_EPE1:
    #     ax.scatter(e_index_DJF[(e_index_DJF>1) & (e_index_DJF<2) & (c_index_DJF>-1) & (c_index_DJF<1)], c_index_DJF[(e_index_DJF>1) & (e_index_DJF<2) & (c_index_DJF>-1) & (c_index_DJF<1)], c='#ee7959', s=5)
    # if num_EPE2:
    #     ax.scatter(e_index_DJF[(e_index_DJF>2) & (c_index_DJF>-1) & (c_index_DJF<1)], c_index_DJF[(e_index_DJF>2) & (c_index_DJF>-1) & (c_index_DJF<1)], c='#ff0000', s=5)
    ax.scatter(e_index_DJF[(e_index_DJF>1) & (c_index_DJF>-1) & (c_index_DJF<1)], c_index_DJF[(e_index_DJF>1) & (c_index_DJF>-1) & (c_index_DJF<1)], c='#FF0000', s=10)
    ax.scatter(e_index_DJF[(e_index_DJF<-1) & (c_index_DJF>-1) & (c_index_DJF<1)], c_index_DJF[(e_index_DJF<-1) & (c_index_DJF>-1) & (c_index_DJF<1)], c='k', s=10) #'#0000ff'

    # ax.text(0.8, 0.62, f"{num_EPE/num_all_DJF:.1%}", fontsize=12, transform=ax.transAxes, ha='left', va='top', color='#ff0000')
    # ax.text(0.07, 0.62, f"{num_EPL/num_all_DJF:.1%}", fontsize=12, transform=ax.transAxes, ha='left', va='top', color='#0000ff')
    # CPE, CPL
    ax.scatter(e_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF>1)], c_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF>1)], c='#e7511b', s=10)
    ax.scatter(e_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF<-1)], c_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF<-1)], c='k', s=10) #'#141475'
    num_CPE = len(e_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF>1)])
    num_CPL = len(e_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF<-1)])
    # ax.text(0.32, 0.9, f"{num_CPE/num_all_DJF:.1%}", fontsize=12, transform=ax.transAxes, ha='left', va='top', color='#910000')
    # ax.text(0.32, 0.2, f"{num_CPL/num_all_DJF:.1%}", fontsize=12, transform=ax.transAxes, ha='left', va='top', color='#141475')
    # EPE&CPE
    ax.scatter(e_index_DJF[(e_index_DJF>1) & (c_index_DJF>1)], c_index_DJF[(e_index_DJF>1) & (c_index_DJF>1)], c='#FFD460', s=10)
    num_mix_nino = len(e_index_DJF[(e_index_DJF>1) & (c_index_DJF>1)])
    # ax.text(0.7, 0.8, f"{num_mix_nino/num_all_DJF:.1%}", fontsize=12, transform=ax.transAxes, ha='left', va='top', color='#ffb5d8')
    # EPL&CPL
    # ax.scatter(e_index_DJF[(e_index_DJF<-1) & (c_index_DJF<-1)], c_index_DJF[(e_index_DJF<-1) & (c_index_DJF<-1)], c='#85d8ff', s=5)
    # num_mix_nina = len(e_index_DJF[(e_index_DJF<-1) & (c_index_DJF<-1)])
    # ax.text(0.1, 0.3, f"{num_mix_nina/num_all_DJF:.1%}", fontsize=12, transform=ax.transAxes, ha='left', va='top', color='#85d8ff')
    # neutral
    ax.scatter(e_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF>-1) & (c_index_DJF<1)], c_index_DJF[(e_index_DJF>-1) & (e_index_DJF<1) & (c_index_DJF>-1) & (c_index_DJF<1)], c='k', s=10)
    # others
    num_of_others = len(e_index_DJF[(e_index_DJF>1) & (c_index_DJF<-1)])
    if num_of_others:
        print("Warning!!!", num_of_others)

    ax.set_xlim(-4, 6)
    ax.set_ylim(-5, 5)
    ax.set_xticks(np.arange(-4, 8, 2))
    ax.set_yticks(np.arange(-6, 6, 2))
    ax.axhline(y=0, color='gray', linestyle='--')
    ax.axvline(x=0, color='gray', linestyle='--')
    ax.axhline(y=1, color='lightsteelblue', linestyle='--', linewidth=1)
    ax.axvline(x=1, color='lightsteelblue', linestyle='--', linewidth=1)
    ax.axhline(y=-1, color='lightsteelblue', linestyle='--', linewidth=1)
    ax.axvline(x=-1, color='lightsteelblue', linestyle='--', linewidth=1)
    # ax.axvline(x=2, color='#5cdbd3', linestyle='--', linewidth=1)
    minor_tick(ax, 2, 2)
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    plt.tick_params(axis='both', which='major', length=6)
    plt.tick_params(axis='both', which='minor', length=4)

    # fig.suptitle(pic_name[pic_name.rfind('/') + 1:], fontsize=16, fontweight='bold')

    customize_font_sizes(ax)
    plt.tight_layout()
    # plt.savefig(pic_name+'.png', dpi=400)
    plt.savefig(pic_name+'.svg')

    return num_EPE2, num_EPE, num_CPL

def draw_scatter1(arr1, arr2, pic_name, label, title, lim=None, tick=None):
    plt.close()
    rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    for i in range(len(arr1)):
        ax.scatter(arr1[i], arr2[i], c='#38c5ba', s=20)
        ax.text(arr1[i], arr2[i]+0.01, i+1, fontsize=9, color = "dimgray", weight="bold", style = "italic", verticalalignment='center', horizontalalignment='center')

    draw_trend(ax, arr1, arr2)

    if lim is not None:
        if lim[0] == "sci":
            ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
            ax.set_xlim(lim[1], lim[2])
            ax.set_ylim(lim[3], lim[4])
        else:
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
    plt.tick_params(axis='both', which='major', length=6)
    plt.tick_params(axis='both', which='minor', length=4)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    customize_font_sizes(ax)
    plt.tight_layout()
    plt.savefig(pic_name, dpi=400)

def cal_CEindex(pc1, pc2):
    eindex = (pc1 - pc2) / (2**0.5)
    cindex = (pc1 + pc2) / (2**0.5)
    return eindex, cindex

def cal_skew(arr):
    from scipy.stats import skew
    skewness = skew(arr, axis=1)
    return skewness

def draw_line(ax, alpha):
    ax.axhline(y=alpha, linestyle='--', linewidth=1, color='k', dashes=(5, 5))
    # ax.axhline(y=alpha/2, linestyle='-.', linewidth=1, color='k')
    # ax.axhline(y=0, linestyle=':', linewidth=1, color='k')
    # ax.axvline(x=0, linestyle=':', linewidth=1, color='k')

def sort_with_indices(numbers):
    # 创建包含（索引，值）的元组列表
    indexed_numbers = list(enumerate(numbers))
    # 按照值从大到小排序
    sorted_indexed_numbers = sorted(indexed_numbers, key=lambda x: x[1], reverse=True)
    # 提取排序后的值和原始索引
    sorted_numbers = [item[1] for item in sorted_indexed_numbers]
    original_indices = [item[0]+1 for item in sorted_indexed_numbers]
    return sorted_numbers, original_indices

if __name__ == '__main__':
    start = time.time()
    print('ᕕ( ᐛ )ᕗ')
    print('==========程序开始运行==========')
    # 读入&处理数据----------------------------------------------------------------
    res_path = "/home/sunming/data5/hongyt/code/sect_eof/res_eof/1920-2005/"
    pc = xr.open_dataarray(res_path+'pcs.nc')
    e_index = xr.open_dataarray(res_path+'e_index.nc')
    c_index = xr.open_dataarray(res_path+'c_index.nc')

    pc1 = pc[:, :, 0]
    pc2 = pc[:, :, 1]
    pc1_DJF = sel_DJF(pc1)
    pc2_DJF = sel_DJF(pc2)
    # e_index_DJF = sel_DJF(e_index)
    # c_index_DJF = sel_DJF(c_index)
    e_index_DJF = DJF_mean(e_index)
    c_index_DJF = DJF_mean(c_index)
    print(e_index_DJF.shape)

    # 修改项, 时间范围
    start_year="1920"
    end_year="2005"
    time_range=start_year+"-"+end_year
    e_index = e_index.loc[:, start_year+"-01-01":end_year+"-12-31"]
    c_index = c_index.loc[:, start_year+"-01-01":end_year+"-12-31"]
    e_index_DJF = e_index_DJF.loc[:, start_year+"-01-01":end_year+"-12-31"]
    c_index_DJF = c_index_DJF.loc[:, start_year+"-01-01":end_year+"-12-31"]
    pc1 = pc1.loc[:, start_year+"-01-01":end_year+"-12-31"]
    pc2 = pc2.loc[:, start_year+"-01-01":end_year+"-12-31"]
    pc1_DJF = pc1_DJF.loc[:, start_year+"-01-01":end_year+"-12-31"]
    pc2_DJF = pc2_DJF.loc[:, start_year+"-01-01":end_year+"-12-31"]
    

    # 画图--------------------------------------------------------------------------

    num_mix_list = []
    num_EN_mix_list = []
    num_LN_mix_list = []
    num_super_list = []
    num_EPE_list = []
    num_CPL_list = []

    draw_path = "/home/sunming/data5/hongyt/code/quadratic_reg/draw_scatter_CE-index_DJF_mean/elnino_only/"
    #draw_sctter_CE-index/"
    for i in range(e_index.shape[0]):
        num_EPE2, num_EPE, num_CPL = draw_scatter(e_index[i], c_index[i], e_index_DJF[i], c_index_DJF[i], draw_path+"r"+str(i+1)+"i1p1", ['E-index', 'C-index'])
        num_super_list.append(num_EPE2)
        num_EPE_list.append(num_EPE)
        num_CPL_list.append(num_CPL)
        e_index1 = e_index_DJF[i]
        c_index1 = c_index_DJF[i]
        num_EN_mix = len(e_index1[(e_index1>1) & (c_index1>1)])
        num_LN_mix = len(e_index1[(e_index1<-1) & (c_index1<-1)])
        print(i+1)
        print("EN mix:", num_EN_mix)
        print("LN mix:", num_LN_mix)
        num_mix = num_EN_mix+num_LN_mix
        num_mix_list.append(num_mix)
        num_EN_mix_list.append(num_EN_mix)
        num_LN_mix_list.append(num_LN_mix)

    # values, index = sort_with_indices(num_super_list)
    # print(values)
    # print(index)
    # num_super = xr.DataArray(num_super_list, dims=('member'), coords={'member': np.arange(1, 41, 1)})
    # num_super.to_dataset(name='num_super').to_netcdf('/home/sunming/data5/hongyt/data/sel_CESM1-LENS/num_super.nc')
    # num_EPE = xr.DataArray(num_EPE_list, dims=('member'), coords={'member': np.arange(1, 41, 1)})
    # num_EPE.to_dataset(name='num_EPE').to_netcdf('/home/sunming/data5/hongyt/data/sel_CESM1-LENS/num_EPE.nc')
    # num_CPL = xr.DataArray(num_CPL_list, dims=('member'), coords={'member': np.arange(1, 41, 1)})
    # num_CPL.to_dataset(name='num_CPL').to_netcdf('/home/sunming/data5/hongyt/data/sel_CESM1-LENS/num_CPL.nc')

    # res_path = "/home/sunming/data5/hongyt/code/quadratic_reg/data/"
    # draw_path = "/home/sunming/data5/hongyt/code/quadratic_reg/draw_confusion_alpha/"
    # alpha = xr.open_dataset(res_path+"alpha_DJF.nc").alpha
    # print(alpha.shape)
    # print(len(num_mix_list))
    # lim1 = [4, 36, -0.6, 0]
    # tick1 = [np.arange(4, 40, 4), np.arange(-0.6, 0.2, 0.2)]
    # draw_scatter1(num_mix_list, alpha, draw_path+"both_confusion.png", ['Num of confusion', 'Alpha'], "El Nino & La Nina", lim1, tick1)
    # lim2 = [0, 24, -0.6, 0]
    # tick2 = [np.arange(0, 28, 4), np.arange(-0.6, 0.2, 0.2)]
    # draw_scatter1(num_EN_mix_list, alpha, draw_path+"EN_confusion.png", ['Num of confusion', 'Alpha'], "El Nino", lim2, tick2)
    # lim3 = [-1, 16, -0.6, 0]
    # tick3 = [np.arange(0, 20, 4), np.arange(-0.6, 0.2, 0.2)]
    # draw_scatter1(num_LN_mix_list, alpha, draw_path+"LN_confusion.png", ['Num of confusion', 'Alpha'], "La Nina", lim3, tick3)


    # # 绘图(skewness/alpha 散点)========================================================
    # sk_e_obs = 1.5377629482212403
    # sk_c_obs = -0.4470542635570995
    # alpha_obs = -0.3070994
    # alpha_DJF_obs = -0.3787121


    print('==========程序运行结束==========')
    print('₍₍٩( ᐛ )۶₎₎♪')
    print('程序共运行了 %f 秒' % (time.time() - start))
