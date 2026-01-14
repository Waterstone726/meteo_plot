import sys
sys.path.append("/home/sunming/data5/hongyt/github/meteo_plot")
from lib.make_ppt import PPTBuilder

# ================= 配置区域 =================

# 基础路径
BASE_DIR = "/home/sunming/data5/hongyt/project/project_lens_to_PI_ToE"
# 模型列表 (用于保证顺序和查找子文件夹)
MODEL_LIST = ["CESM1", "GFDL-ESM2M", "MPI-ESM", "CESM2", "MPI-ESM1-2-LR", "FGOALS-g3"]

# 定义任务列表
# 这里的逻辑完全复刻了你原来的代码 1 和 代码 2 的需求
tasks = [
    # -------------------------------------------------------
    # 场景 1: 代码1 中的 "Index 71 Window" (每页2张图，左右并列)
    # -------------------------------------------------------
    {
        'section_name': "Part 1: ToE Index (Window 71)", 
        'slide_title': "ToE of EC Index (71yr)",
        'base_dir': BASE_DIR,
        'folder': "res_toe_EC_index_std_long",
        'models': MODEL_LIST,           # 按模型顺序查找
        'patterns': ["index_std_71"],   # 文件名必须包含此字符串
        'cols': 2,                      # 2列
        'rows': 1                       # 1行 -> 实现左右并列
    },

    # -------------------------------------------------------
    # 场景 2: 代码1 中的 "2D Pattern" (每页16张图，4x4网格)
    # -------------------------------------------------------
    # 注意：这里我们让程序自动把所有模型的图收集起来
    # 如果你想“一页只放一个模型的16张图”，需要在下面分开写，或者稍微修改逻辑
    # 下面演示的是：把所有找到的图按 4x4 填入，填满一页自动换下一页
    {
        'section_name': "Part 2: 2D Pattern Summary",
        'slide_title': "2D Pattern (4x4 Grid)",
        'base_dir': BASE_DIR,
        'folder': "res_ToE_of_EC_2d_pattern",
        'models': MODEL_LIST,
        'patterns': ["ToE", "reg"], # 宽松匹配，只要包含这些即可 (需根据实际文件名调整)
        # 如果你想严格复刻代码1的“只找特定变量”，可以在 patterns 里写的更具体
        # 或者依靠 search_images 的逻辑
        'cols': 4,
        'rows': 4
    },

    # -------------------------------------------------------
    # 场景 3: 代码2 中的通用排版 (如 E-index Trend, 3x2 排布)
    # -------------------------------------------------------
    {
        'section_name': "Part 3: E-index Trend Spread",
        'slide_title': "Trend Analysis",
        'base_dir': BASE_DIR,
        'folder': "res_trend_of_lens_spread",
        'models': MODEL_LIST,
        'patterns': ["E-index"],
        'cols': 3,
        'rows': 2
    },
    
    {
        'section_name': "Part 4: Scatter Plots",
        'slide_title': "Scatter Analysis",
        'base_dir': BASE_DIR,
        'folder': "res_EC_index_scatter",
        'models': MODEL_LIST,
        'patterns': ["scatter"],
        'cols': 3,
        'rows': 2
    }
]

# ================= 执行 =================

if __name__ == "__main__":
    output_file = "./ppt/Final_Universal_Report.pptx"
    
    # 1. 实例化
    tool = PPTBuilder(output_file)
    
    # 2. 运行任务
    tool.run_tasks(tasks)
