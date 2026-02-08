import pandas as pd
import numpy as np
from scipy.stats import linregress
import os

def save_corr_to_excel(arr1, arr2, file_path, label=("X", "Y"), data_labels=None):
    """
    XlsxWriter 动态公式版：
    1. 统计指标 (N, a, b, R, P, Sig) 全部写入 Excel 公式。
    2. 图表标题动态关联单元格 (修改数据后，图表标题上的 R 值也会变)。
    """
    # 1. 路径与数据准备
    file_path = str(file_path)
    if not file_path.endswith('.xlsx'): file_path += '.xlsx'

    x_raw = np.array(arr1).flatten()
    y_raw = np.array(arr2).flatten()
    length = len(x_raw)

    # 标签准备
    if data_labels is True:
        l_raw = np.arange(1, length + 1)
    elif data_labels is not None and data_labels is not False:
        l_raw = np.array(data_labels).flatten()
        if len(l_raw) < length:
            l_raw = np.pad(l_raw, (0, length - len(l_raw)), constant_values='')
        elif len(l_raw) > length:
            l_raw = l_raw[:length]
    else:
        l_raw = np.full(length, '')

    # 清洗空值 (用于初始数据写入，后续用户可以在Excel里改)
    df = pd.DataFrame({'Label': l_raw, label[0]: x_raw, label[1]: y_raw})
    df[label[0]] = pd.to_numeric(df[label[0]], errors='coerce')
    df[label[1]] = pd.to_numeric(df[label[1]], errors='coerce')
    df_clean = df.dropna(subset=[label[0], label[1]])
    
    # 原始数据行数（用于确定Excel公式的范围）
    # Excel中数据从第2行开始(Index 1)，到 len(df_clean)+1 行
    data_len = len(df_clean)
    excel_max_row = data_len + 1 
    
    # 定义 Excel 中的数据范围字符串，例如 "B2:B10"
    # 假设 Label在A列, X在B列, Y在C列
    range_x = f"B2:B{excel_max_row}"
    range_y = f"C2:C{excel_max_row}"

    # =========================================================
    # 开始绘图 (XlsxWriter)
    # =========================================================
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        sheet_name = 'Analysis'
        
        # 写入初始数据
        df_clean.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)
        
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]
        
        # 样式定义
        header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#4F81BD', 'align': 'center'})
        num_fmt = workbook.add_format({'num_format': '0.00', 'align': 'right'}) # 数字格式
        txt_fmt = workbook.add_format({'align': 'right'})
        p_fmt = workbook.add_format({'num_format': '0.0000', 'align': 'right'}) # P值多几位小数

        # --- 写入统计指标的公式 ---
        # 我们将指标放在第 E 列 (索引4) 和 F 列 (索引5)
        stat_label_col = 4
        stat_val_col = 5
        
        # 1. N (样本量) -> F2
        worksheet.write(1, stat_label_col, "N", header_fmt)
        worksheet.write_formula(1, stat_val_col, f'=COUNT({range_x})', num_fmt)
        
        # 2. a (斜率 Slope) -> F3
        worksheet.write(2, stat_label_col, "a", header_fmt)
        worksheet.write_formula(2, stat_val_col, f'=SLOPE({range_y}, {range_x})', num_fmt)
        
        # 3. b (截距 Intercept) -> F4
        worksheet.write(3, stat_label_col, "b", header_fmt)
        worksheet.write_formula(3, stat_val_col, f'=INTERCEPT({range_y}, {range_x})', num_fmt)
        
        # 4. R (相关系数) -> F5
        worksheet.write(4, stat_label_col, "R", header_fmt)
        worksheet.write_formula(4, stat_val_col, f'=CORREL({range_y}, {range_x})', num_fmt)
        
        # 5. P (P-value) 【核心修复点】
        # 错误原因：T.DIST.2T 是新版函数，旧版或WPS不识别。
        # 修复方案：改用 TDIST (经典兼容函数)。
        # 语法：TDIST(x, degrees_of_freedom, tails) -> tails=2 (双尾)
        worksheet.write(5, stat_label_col, "P", header_fmt)
        
        # 构造 T 统计量的计算部分: |R * sqrt(N-2) / sqrt(1-R^2)|
        t_stat_calc = "ABS(F5*SQRT((F2-2)/(1-F5^2)))"
        
        # 完整公式：如果是 R=1 防止分母为0，否则计算 TDIST
        p_formula = f'=IF(ABS(F5)>0.99999, 0, TDIST({t_stat_calc}, F2-2, 2))'
        
        worksheet.write_formula(5, stat_val_col, p_formula, p_fmt)
        
        # 6. Sig (显著性) 【代码无需改动，P值修复后这里自动恢复】
        worksheet.write(6, stat_label_col, "Sig", header_fmt)
        worksheet.write_formula(6, stat_val_col, '=IF(F6<0.01, "**", IF(F6<0.05, "*", ""))', txt_fmt)

        # 7. 动态标题辅助单元格 【代码无需改动，P值修复后这里自动恢复】
        # 这里的 & 符号用于拼接字符串
        title_formula = '="R = "&TEXT(F5,"0.00")&F7'
        worksheet.write_formula(7, stat_val_col, title_formula, txt_fmt)

        # --- 图表设置 ---
        chart = workbook.add_chart({'type': 'scatter'})
        
        # 准备自定义标签 (注意：如果数据行数变了，这里的静态标签不会变，
        # 但完全动态标签需要非常复杂的公式或VBA，这里保留静态标签写入仅作初始展示)
        custom_labels = []
        labels_valid = df_clean['Label'].values
        if len(labels_valid) > 0:
            for txt in labels_valid:
                custom_labels.append({'value': str(txt)})

        chart.add_series({
            'name':       'Data',
            'categories': [sheet_name, 1, 1, data_len, 1], # B列
            'values':     [sheet_name, 1, 2, data_len, 2], # C列
            'marker': {
                'type': 'circle', 
                'size': 6,
                'fill': {'color': '#4472C4'},
                'border': {'color': 'white', 'width': 0.75}
            },
            'data_labels': {
                'value': False,
                'custom': custom_labels,
                'position': 'top',
                'font': {'color': '#333333', 'size': 8}
            },
            'trendline': {
                'type': 'linear',
                'display_equation': True,
                'display_r_squared': True, # Excel趋势线只能显示R平方，不能显示公式计算的R
                'line': {'color': 'black', 'width': 1.25, 'dash_type': 'solid'}
            }
        })

        # --- 美化与动态设置 ---
        chart.set_size({'width': 800, 'height': 500})

        # 【核心修改】动态图表标题
        # 语法：[sheet_name, row, col] -> 引用 F8 单元格 (索引为 7, 5)
        chart.set_title({
            'name': [sheet_name, 7, 5], 
            'name_font': {'size': 14, 'bold': True}
        })

        chart.set_x_axis({
            'name': label[0],
            'name_font': {'size': 11, 'bold': True},
            'num_font':  {'size': 10},
            'crossing': 'min', 
            'major_gridlines': {'visible': True, 'line': {'color': '#D9D9D9'}},
            'line': {'color': 'black'}
        })

        chart.set_y_axis({
            'name': label[1],
            'name_font': {'size': 11, 'bold': True},
            'num_font':  {'size': 10},
            'major_gridlines': {'visible': True, 'line': {'color': '#D9D9D9'}},
            'line': {'color': 'black'}
        })

        chart.set_legend({'none': True})
        chart.set_plotarea({
            'border': {'none': True},
            'fill':   {'color': 'white'}
        })

        worksheet.insert_chart('H2', chart)

    print(f"Excel saved: {file_path}")

# --- 测试 ---
if __name__ == "__main__":
    labels = [1, 2, 3, 4, 11, 12, 13]
    x = [-0.3, -0.22, -0.05, -0.24, -0.30, -0.15, -0.10]
    y = [1961, 2023, 2023, 1999, np.nan, 2050, 2010]
    
    save_corr_to_excel(x, y, "Data_Chart.xlsx", label=("X_Value", "Y_Value"), data_labels=labels)