import pandas as pd
import numpy as np
import os


def process_wear_data(wear_df):
    """处理磨损数据，计算每次掘进段的磨损差值，并处理换刀情况"""
    # 复制原始数据用于计算差值
    wear_copy = wear_df.copy()

    # 对每把刀具计算与上一段的差值
    for col in wear_df.columns:
        if col.startswith('cutter_'):
            # 计算当前段与上一段的差值
            wear_df[col] = wear_df[col] - wear_copy[col].shift(1)

            # 处理换刀情况（如果差值为负，说明换刀了，当前磨损就是绝对值）
            wear_df.loc[wear_df[col] < 0, col] = wear_copy[col][wear_df[col] < 0]

    # 第一段没有前序数据，直接使用原始值
    first_segment = wear_df.index[0]
    for col in wear_df.columns:
        if col.startswith('cutter_'):
            wear_df.loc[first_segment, col] = wear_copy.loc[first_segment, col]

    return wear_df


def aggregate_tunnel_data(tunnel_df):
    """对掘进数据进行聚合，计算每个区间段的统计特征"""
    # 重命名列名为英文
    tunnel_rename_map = {
        '掘进位移(m)': 'tunneling_displacement',
        '刀盘转速（RPM）': 'cutter_head_speed_rpm',
        '主推进力（kN）': 'main_thrust_kn',
        '扭矩压力（bar）': 'torque_pressure_bar',
        '刀盘扭矩（kN·m）': 'cutter_head_torque_kNm',
        '辅助夹持器压力（bar）': 'auxiliary_gripper_pressure_bar',
        '辅助推进油缸 1': 'auxiliary_thrust_cylinder_1',
        '辅助推进油缸 2': 'auxiliary_thrust_cylinder_2',
        '辅助推进油缸 3': 'auxiliary_thrust_cylinder_3',
        '吸收泵 M1 电流': 'absorption_pump_m1_current',
        '吸收泵 M2 电流': 'absorption_pump_m2_current',
        '吸收泵 M3 电流': 'absorption_pump_m3_current',
        '吸收泵 M4 电流': 'absorption_pump_m4_current',
        '吸收泵 M5 电流': 'absorption_pump_m5_current',
        '吸收泵 M6 电流': 'absorption_pump_m6_current',
        '油液湿度传感器': 'oil_humidity_sensor',
        '液压油位': 'hydraulic_oil_level',
        '润滑压力 PTX97_05': 'lubrication_pressure_ptx97_05',
        '润滑压力 PTX97_06': 'lubrication_pressure_ptx97_06',
        '润滑压力 PTX_97_03': 'lubrication_pressure_ptx97_03',
        '主推进油缸 1 和 10': 'main_thrust_cylinder_1_10',
        '主推进油缸 2 和 3': 'main_thrust_cylinder_2_3',
        '主推进油缸 4 和 5': 'main_thrust_cylinder_4_5',
        '主推进油缸 6 和 7': 'main_thrust_cylinder_6_7',
        '主推进油缸 8 和 9': 'main_thrust_cylinder_8_9',
        '主推进活塞杆 1': 'main_thrust_piston_rod_1',
        '主推进活塞杆 10': 'main_thrust_piston_rod_10',
        '闭路压水系统状态': 'closed_loop_water_system_status',
        '备用回转输送压力': 'backup_rotary_conveying_pressure',
        '盾构机回转输送压力': 'tbm_rotary_conveying_pressure',
        '推进比例阀开度 - 下': 'thrust_proportion_valve_bottom',
        '推进比例阀开度 - 左': 'thrust_proportion_valve_left',
        '推进比例阀开度 - 右': 'thrust_proportion_valve_right',
        '推进比例阀开度 - 上': 'thrust_proportion_valve_top',
        '回油压力': 'return_oil_pressure',
        '滚动盾构夹持器状态': 'rolling_shield_gripper_status',
        '稳定器压力（bar）': 'stabilizer_pressure_bar',
        '闭路水温': 'closed_loop_water_temperature',
        '扭矩电机 01': 'torque_motor_01',
        '扭矩电机 02': 'torque_motor_02',
        '扭矩电机 03': 'torque_motor_03',
        '扭矩电机 04': 'torque_motor_04',
        '扭矩电机 07': 'torque_motor_07',
        '推进速度 AR（mm/min）': 'advance_speed_mm_per_min'
    }
    tunnel_df = tunnel_df.rename(columns=tunnel_rename_map)

    # 对掘进距离进行四舍五入到最近的1.5m倍数，确定所属区间段
    tunnel_df['distance_segment'] = np.round(tunnel_df['tunneling_displacement'] / 1.5) * 1.5

    # 定义需要聚合的列和对应的聚合函数
    # 对于大多数参数，我们计算均值、最大值、最小值、标准差和范围
    agg_dict = {}
    for col in tunnel_df.columns:
        if col not in ['tunneling_displacement', 'distance_segment']:
            # 推进速度作为预测值，计算其均值、最终值和变化趋势
            if col == 'advance_speed_mm_per_min':
                agg_dict[col] = ['mean', 'max', 'min', 'last', lambda x: x.iloc[-1] - x.iloc[0]]
            else:
                agg_dict[col] = ['mean', 'max', 'min', 'std', lambda x: x.max() - x.min()]

    # 按区间分组，计算统计特征
    tunnel_aggregated = tunnel_df.groupby('distance_segment').agg(agg_dict)

    # 展平列名
    tunnel_aggregated.columns = ['_'.join(col).strip().replace('<lambda_0>', 'range').replace('<lambda_1>', 'trend')
                                 for col in tunnel_aggregated.columns.values]

    # 重置索引，使distance_segment成为普通列
    tunnel_aggregated = tunnel_aggregated.reset_index()

    return tunnel_aggregated


def merge_tunnel_data(cutter_layout_path, geologic_path, tunnel_data_path, wear_data_path, output_path):
    """合并四个Excel文件的数据"""
    print("正在读取数据文件...")
    cutter_df = pd.read_excel(cutter_layout_path)
    geologic_df = pd.read_excel(geologic_path)
    tunnel_df = pd.read_excel(tunnel_data_path)
    wear_df = pd.read_excel(wear_data_path)

    print("正在预处理数据...")

    cutter_df.columns = ['cutter_id', 'radius_mm', 'type', 'installation_plane_angle',
                         'inclination_to_axis', 'wear_limit_mm', 'neighbors']

    wear_df.columns = ['segment_id', 'tunneling_distance', 'energy',
                       *[f'cutter_{i}_wear' for i in range(1, 42)]]
    wear_df = process_wear_data(wear_df)

    tunnel_aggregated = aggregate_tunnel_data(tunnel_df)

    geologic_df.columns = ['segment_id', 'ucs_mpa', 'cai', 'bts_mpa', 'des_g_per_mm2', 'bq']

    segment_distance_map = wear_df[['segment_id', 'tunneling_distance']].set_index('segment_id').to_dict()[
        'tunneling_distance']

    geologic_df['tunneling_distance'] = geologic_df['segment_id'].map(segment_distance_map)

    print("正在合并数据...")

    merged_df = pd.merge(geologic_df, wear_df, on=['segment_id', 'tunneling_distance'], how='inner')

    merged_df = pd.merge(merged_df, tunnel_aggregated,
                         left_on='tunneling_distance', right_on='distance_segment',
                         how='left')

    for cutter_id in cutter_df['cutter_id'].unique():
        cutter_info = cutter_df[cutter_df['cutter_id'] == cutter_id].iloc[0].to_dict()
        for key, value in cutter_info.items():
            if key != 'cutter_id':
                merged_df[f'cutter_{cutter_id}_{key}'] = value

    merged_df.to_excel(output_path, index=False)
    print(f"数据合并完成，已保存至 {output_path}")

    return merged_df


if __name__ == "__main__":
    cutter_layout_path = "data/Cutter_layout_template.xlsx"
    geologic_path = "data/geologic_parameter.xlsx"
    tunnel_data_path = "data/tunnel_data.xlsx"
    wear_data_path = "data/wear_data.xlsx"
    output_path = "data/merged_tunnel_data.xlsx"

    for path in [cutter_layout_path, geologic_path, tunnel_data_path, wear_data_path]:
        if not os.path.exists(path):
            print(f"错误：文件 {path} 不存在，请检查路径是否正确。")
            exit(1)

    # 合并数据
    merged_data = merge_tunnel_data(
        cutter_layout_path,
        geologic_path,
        tunnel_data_path,
        wear_data_path,
        output_path
    )

    print(f"合并后的数据集包含 {merged_data.shape[0]} 行和 {merged_data.shape[1]} 列")

    prediction_columns = [col for col in merged_data.columns if
                          col.startswith('cutter_') and col.endswith('_wear') or
                          col == 'energy' or
                          col.startswith('advance_speed_mm_per_min')]
    print("\n预测值字段包括：")
    for col in prediction_columns:
        print(f"- {col}")
