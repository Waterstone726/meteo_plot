import pandas as pd
import numpy as np
import os
from xlsxwriter.utility import xl_rowcol_to_cell

def _write_analysis_block(workbook, worksheet, start_row, start_col, df, label_names=("X", "Y")):
    """
    内部核心函数：在指定位置写入一套完整的 [数据 + 统计公式 + 交互图表]
    """
    data_len = len(df)
    
    # 1. 样式定义 (复用 workbook)
    header_fmt = workbook.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#4F81BD', 'align': 'center'})
    num_fmt = workbook.add_format({'num_format': '0.00', 'align': 'right'})
    txt_fmt = workbook.add_format({'align': 'right'})
    p_fmt = workbook.add_format({'num_format': '0.0000', 'align': 'right'})

    # 2. 写入表头
    headers = ['Label', label_names[0], label_names[1]]
    worksheet.write_row(start_row, start_col, headers, header_fmt)
    
    # 3. 写入数据体 (逐行写入，避免Pandas覆盖格式)
    for i in range(data_len):
        r = start_row + 1 + i
        worksheet.write(r, start_col,     df.iloc[i, 0])      # Label
        worksheet.write(r, start_col + 1, df.iloc[i, 1])      # X
        worksheet.write(r, start_col + 2, df.iloc[i, 2])      # Y

    # 4. 获取动态范围字符串 (关键：相对引用)
    # X列范围: (start_col + 1)
    cell_x_start = xl_rowcol_to_cell(start_row + 1, start_col + 1)
    cell_x_end   = xl_rowcol_to_cell(start_row + data_len, start_col + 1)
    range_x = f"{cell_x_start}:{cell_x_end}"
    
    # Y列范围: (start_col + 2)
    cell_y_start = xl_rowcol_to_cell(start_row + 1, start_col + 2)
    cell_y_end   = xl_rowcol_to_cell(start_row + data_len, start_col + 2)
    range_y = f"{cell_y_start}:{cell_y_end}"

    # 5. 写入统计指标 (在数据右侧空一列: start_col + 4)
    stat_label_col = start_col + 4
    stat_val_col   = start_col + 5
    curr_row = start_row + 1

    # N
    worksheet.write(curr_row, stat_label_col, "N", header_fmt)
    worksheet.write_formula(curr_row, stat_val_col, f'=COUNT({range_x})', num_fmt)
    addr_N = xl_rowcol_to_cell(curr_row, stat_val_col) 

    # a (Slope)
    curr_row += 1
    worksheet.write(curr_row, stat_label_col, "a", header_fmt)
    worksheet.write_formula(curr_row, stat_val_col, f'=SLOPE({range_y}, {range_x})', num_fmt)

    # b (Intercept)
    curr_row += 1
    worksheet.write(curr_row, stat_label_col, "b", header_fmt)
    worksheet.write_formula(curr_row, stat_val_col, f'=INTERCEPT({range_y}, {range_x})', num_fmt)

    # R (Correl)
    curr_row += 1
    worksheet.write(curr_row, stat_label_col, "R", header_fmt)
    worksheet.write_formula(curr_row, stat_val_col, f'=CORREL({range_y}, {range_x})', num_fmt)
    addr_R = xl_rowcol_to_cell(curr_row, stat_val_col) 

    # P (兼容性修复: TDIST)
    curr_row += 1
    worksheet.write(curr_row, stat_label_col, "P", header_fmt)
    t_stat_calc = f"ABS({addr_R}*SQRT(({addr_N}-2)/(1-{addr_R}^2)))"
    # 增加容错：若R=1，T无限大，P为0
    p_formula = f'=IF(ABS({addr_R})>0.99999, 0, TDIST({t_stat_calc}, {addr_N}-2, 2))'
    worksheet.write_formula(curr_row, stat_val_col, p_formula, p_fmt)
    addr_P = xl_rowcol_to_cell(curr_row, stat_val_col)

    # Sig
    curr_row += 1
    worksheet.write(curr_row, stat_label_col, "Sig", header_fmt)
    worksheet.write_formula(curr_row, stat_val_col, f'=IF({addr_P}<0.01, "**", IF({addr_P}<0.05, "*", ""))', txt_fmt)
    addr_Sig = xl_rowcol_to_cell(curr_row, stat_val_col)

    # Title Helper (辅助单元格)
    curr_row += 1
    worksheet.write_formula(curr_row, stat_val_col, f'="R = "&TEXT({addr_R},"0.00")&{addr_Sig}', txt_fmt)

    # 6. 图表设置
    chart = workbook.add_chart({'type': 'scatter'})
    sheet_name = worksheet.get_name()

    # 初始标签 (静态)
    custom_labels = []
    labels_valid = df['Label'].values
    for txt in labels_valid:
        custom_labels.append({'value': str(txt)})

    chart.add_series({
        'name':       'Data',
        'categories': [sheet_name, start_row + 1, start_col + 1, start_row + data_len, start_col + 1],
        'values':     [sheet_name, start_row + 1, start_col + 2, start_row + data_len, start_col + 2],
        'marker':     {'type': 'circle', 'size': 6, 'fill': {'color': '#4472C4'}, 'border': {'color': 'white'}},
        'data_labels': {'value': False, 'custom': custom_labels, 'position': 'top', 'font': {'color': '#333333', 'size': 8}},
        'trendline':  {'type': 'linear', 'display_equation': True, 'display_r_squared': True, 'line': {'color': 'black'}}
    })

    # 图表位置与美化
    chart.set_size({'width': 600, 'height': 400})
    chart.set_title({'name': [sheet_name, curr_row, stat_val_col], 'name_font': {'size': 14, 'bold': True}})
    chart.set_legend({'none': True})
    chart.set_plotarea({'border': {'none': True}, 'fill': {'color': 'white'}})
    chart.set_x_axis({'name': label_names[0], 'crossing': 'min', 'major_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}}, 'line': {'color': 'black'}})
    chart.set_y_axis({'name': label_names[1], 'major_gridlines': {'visible': True, 'line': {'color': '#E0E0E0'}}, 'line': {'color': 'black'}})

    # 插入图表到数据右侧
    chart_pos = xl_rowcol_to_cell(start_row, start_col + 7)
    worksheet.insert_chart(chart_pos, chart)


def save_corr_excel_comparison(arr1, arr2, file_path, label=("X", "Y"), data_labels=None):
    """
    通用版：输入一份数据，生成两个完全一样的分析块（左侧为原始，右侧供修改）。
    参数与原函数保持一致，无需传入两组数据。
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

    # 清洗空值 (统一处理)
    df = pd.DataFrame({'Label': l_raw, label[0]: x_raw, label[1]: y_raw})
    df[label[0]] = pd.to_numeric(df[label[0]], errors='coerce')
    df[label[1]] = pd.to_numeric(df[label[1]], errors='coerce')
    df_clean = df.dropna(subset=[label[0], label[1]])
    
    if len(df_clean) < 2:
        print("Error: Not enough valid data points.")
        return

    # =========================================================
    # 开始生成 Excel
    # =========================================================
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        workbook = writer.book
        sheet_name = 'Comparison_Analysis'
        worksheet = workbook.add_worksheet(sheet_name)
        
        # --- 模块1：原始数据 (左侧，从 A1 开始) ---
        print("Generating Block 1 (Original)...")
        _write_analysis_block(
            workbook, worksheet, 
            start_row=0, start_col=0, 
            df=df_clean, 
            label_names=label
        )

        # --- 模块2：对照副本 (右侧，从 Q1 开始) ---
        # 使用相同的数据 df_clean，生成一份完全独立的副本
        print("Generating Block 2 (Editable Copy)...")
        _write_analysis_block(
            workbook, worksheet, 
            start_row=0, start_col=16, # Q列开始 (中间隔开足够距离)
            df=df_clean, 
            label_names=label
        )

    print(f"Excel saved (Dual Blocks): {file_path}")

# --- 测试 ---
if __name__ == "__main__":
    # 你的原始数据
    labels = [1, 2, 3, 4, 11, 12, 13]
    x = [-0.3, -0.22, -0.05, -0.24, -0.30, -0.15, -0.10]
    y = [1961, 2023, 2023, 1999, np.nan, 2050, 2010]
    
    # 调用函数 (参数完全没变)
    save_corr_excel_comparison(x, y, "My_Experiment.xlsx", label=("X_Val", "Y_Val"), data_labels=labels)