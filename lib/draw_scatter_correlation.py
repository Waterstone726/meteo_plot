import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, t
from matplotlib import rc
from matplotlib.ticker import AutoMinorLocator

# ==========================================
# 基础风格辅助函数 (保留你原本的所有设置)
# ==========================================

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

def minor_tick(ax, num1, num2):
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=num1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=num2))

def get_sig_stars(p_value):
    """根据显著性返回星号"""
    if p_value < 0.01: return "**"
    elif p_value < 0.05: return "*"
    return ""

# ==========================================
# 核心解耦函数：负责在指定的 ax 上绘图
# ==========================================

def draw_scatter_correlation_single(ax, arr1, arr2, label=("X", "Y"), title="", 
                                   order_text=False, draw_ci=True, **kwargs):
    """
    针对单个 Axes 的绘图逻辑
    **kwargs: 接收 scatter 的各种参数，如 s, c, alpha, marker 等
    """
    # 1. 解决 NaN 问题：联合掩码
    x, y = np.array(arr1), np.array(arr2)
    mask = np.isfinite(x) & np.isfinite(y)
    x_c, y_c = x[mask], y[mask]

    # 2. 统计计算 (显著性检验)
    if len(x_c) > 1:
        slope, intercept, r_value, p_value, std_err = linregress(x_c, y_c)
        stars = get_sig_stars(p_value)
        print(p_value)
        
        # 拟合直线
        x_range = np.linspace(np.min(x_c), np.max(x_c), 100)
        y_fit = slope * x_range + intercept
        ax.plot(x_range, y_fit, color="#b82c25", zorder=2, linewidth=1.5)

        # 3. 置信区间 (CI)
        if draw_ci:
            t_inv = t.ppf(0.975, len(x_c) - 2)
            resid = y_c - (slope * x_c + intercept)
            s_err = np.sqrt(np.sum(resid**2) / (len(x_c) - 2))
            ci = t_inv * s_err * np.sqrt(1/len(x_c) + (x_range - np.mean(x_c))**2 / np.sum((x_c - np.mean(x_c))**2))
            ax.fill_between(x_range, y_fit - ci, y_fit + ci, color="#b82c25", alpha=0.15, edgecolor='none', zorder=1)

        # 4. 文本布局 (灵活位置 + 星号标注)
        info_text = f"R = {r_value:.2f}{stars}\nSlope = {slope:.2e}{stars}"
        # 默认放在左上角，handlelength=0 保留了你原本 add_text 的 legend 技巧感
        ax.legend([info_text], loc="upper left", handlelength=0, handletextpad=0, 
                  fontsize=11, frameon=False)

    # 5. 散点质感优化 (保留原本颜色，增加边缘色)
    scatter_defaults = {'s': 20, 'c': '#a7a7a7', 'edgecolors': 'white', 'linewidths': 0.5, 'alpha': 0.8, 'zorder': 3}
    if order_text: scatter_defaults['c'] = "#22a6be"
    scatter_defaults.update(kwargs) # 允许外部覆盖
    
    ax.scatter(x, y, **scatter_defaults)

    # 6. 动态 Offset 标注数字
    if order_text:
        y_span = np.nanmax(y) - np.nanmin(y)
        offset = y_span * 0.02
        for i in range(len(x)):
            if np.isfinite(x[i]) and np.isfinite(y[i]):
                ax.text(x[i], y[i] + offset, i+1, fontsize=9, color="#6D6C6C", 
                        weight="bold", style="italic", ha='center', va='bottom')

    # 7. 基础修饰 (继承你原本的要求)
    ax.set_xlabel(label[0], fontweight='bold')
    ax.set_ylabel(label[1], fontweight='bold')
    ax.set_title(title)
    minor_tick(ax, 2, 2)
    customize_font_sizes(ax)

# ==========================================
# 原函数：保持接口不变，负责整体风格和文件保存
# ==========================================

def draw_scatter_correlation(arr1, arr2, pic_name, label, title, order_text=False, lim=None, tick=None):
    """
    保留原本的接口，确保旧代码不报错
    """
    plt.close()
    # --- 你的原始风格设定 ---
    rc('font', family='Times New Roman', weight='semibold')
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.titlepad'] = 30
    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    
    # 粗边框设置
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(True)  
        ax.spines[spine].set_linewidth(2)
    # -----------------------

    # 调用解耦后的单图绘制函数
    draw_scatter_correlation_single(ax, arr1, arr2, label=label, title=title, order_text=order_text)

    # 处理原有的 lim 和 tick 参数
    if lim is not None:
        ax.set_xlim(lim[0], lim[1])
        ax.set_ylim(lim[2], lim[3])
    if tick is not None:
        ax.set_xticks(tick[0])
        ax.set_yticks(tick[1])

    plt.tight_layout()
    plt.savefig(pic_name, dpi=400)
    print(f"Successfully saved to {pic_name}")

if __name__ == '__main__':
    # 测试代码
    x = np.random.randn(20)
    y = x * 0.7 + np.random.randn(20) * 0.3
    # 原接口调用方式，完全兼容
    draw_scatter_correlation(x, y, "test_output.png", ["Index X", "Index Y"], "Sample Correlation")