import pandas as pd


def process_wear_data(wear_df):
    """
    处理磨损数据，计算磨损增量并处理负值

    参数:
        wear_df: 包含磨损数据的DataFrame，列名以'wear_'开头

    返回:
        处理后的磨损数据，包含原始磨损量(添加_total后缀)和磨损增量(添加_increment后缀)
    """
    # 创建数据副本以避免修改原始数据
    wear_copy = wear_df.copy()
    # 识别所有磨损相关列
    wear_columns = [col for col in wear_df.columns if col.startswith('wear_')]

    for col in wear_columns:
        # 计算磨损增量（当前值减去前一行值）
        wear_df[f'{col}_increment'] = wear_df[col] - wear_copy[col].shift(1)
        # 第一行的增量等于其本身值
        wear_df.loc[0, f'{col}_increment'] = wear_df.loc[0, col]
        # 处理负增量（可能由测量误差引起）
        negative_mask = wear_df[f'{col}_increment'] < 0
        if negative_mask.any():
            wear_df.loc[negative_mask, f'{col}_increment'] = 0
        # 重命名原始磨损列为总磨损
        wear_df.rename(columns={col: f'{col}_total'}, inplace=True)

    return wear_df


def aggregate_tunnel_data(tunnel_df, wear_df):
    """
    聚合隧道掘进机操作数据，按距离段计算统计特征

    参数:
        tunnel_df: 包含隧道掘进机操作数据的DataFrame
        wear_df: 包含磨损数据的DataFrame，用于确定距离分段

    返回:
        按距离段聚合后的隧道操作数据，包含各特征的均值、标准差、最大值和最小值
    """
    # 移除无名列和全空列
    tunnel_df = tunnel_df.loc[:, ~tunnel_df.columns.str.contains('Unnamed', na=False)]
    tunnel_df = tunnel_df.dropna(axis=1, how='all')

    # 重命名列名，使其更规范
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
        '推进速度 AR（mm/min） ': 'advance_speed_mm_per_min'
    }
    tunnel_df = tunnel_df.rename(columns=tunnel_rename_map)

    # 移除重复或不需要的列
    if 'cutter_head_torque_kNm' in tunnel_df.columns:
        tunnel_df = tunnel_df.drop(columns=['cutter_head_torque_kNm'])

    # 确保隧道数据不超过磨损数据的最大距离
    max_distance = wear_df['simluate_distance'].max()
    tunnel_df = tunnel_df[tunnel_df['tunneling_displacement'] <= max_distance]

    # 将隧道数据按磨损数据中的距离分段进行分组
    distance_segments = sorted(wear_df['simluate_distance'].unique())
    tunnel_df['distance_segment'] = pd.cut(
        tunnel_df['tunneling_displacement'],
        bins=[0] + distance_segments,
        labels=distance_segments
    ).astype(float)

    # 定义聚合函数：对每个特征计算均值、标准差、最大值和最小值
    agg_dict = {}
    for col in tunnel_df.columns:
        if col not in ['tunneling_displacement', 'distance_segment']:
            agg_dict[col] = ['mean', 'std', 'max', 'min']

    # 按距离段进行聚合
    tunnel_aggregated = tunnel_df.groupby('distance_segment').agg(agg_dict)

    # 重命名聚合后的列，格式为"特征_统计量"
    tunnel_aggregated.columns = [
        f"{col[0]}_{col[1]}"
        for col in tunnel_aggregated.columns.values
    ]

    # 重置索引并规范化列名
    tunnel_aggregated = tunnel_aggregated.reset_index()
    tunnel_aggregated.rename(columns={'distance_segment': 'simluate_distance'}, inplace=True)

    return tunnel_aggregated


def merge_tunnel_data(cutter_layout_path, tunnel_data_path, wear_data_path, output_path):
    """
    合并刀具布局数据、隧道操作数据和磨损数据

    参数:
        cutter_layout_path: 刀具布局数据文件路径
        tunnel_data_path: 隧道操作数据文件路径
        wear_data_path: 磨损数据文件路径
        output_path: 合并后数据的保存路径

    返回:
        合并后的DataFrame
    """
    # 加载各类数据
    cutter_df = pd.read_excel(cutter_layout_path)
    tunnel_df = pd.read_excel(tunnel_data_path)
    wear_df = pd.read_excel(wear_data_path)

    # 处理磨损数据
    wear_df = process_wear_data(wear_df)

    # 规范化刀具布局数据列名
    cutter_df.columns = ['cutter_id', 'radius_mm', 'type', 'installation_plane_angle',
                         'inclination_to_axis', 'wear_limit_mm', 'neighbors']

    # 聚合隧道操作数据
    tunnel_aggregated = aggregate_tunnel_data(tunnel_df, wear_df)

    # 合并磨损数据和聚合后的隧道数据
    merged_df = pd.merge(
        wear_df,
        tunnel_aggregated,
        on='simluate_distance',  # 按距离进行合并
        how='left'
    )

    # 保存合并结果
    merged_df.to_excel(output_path, index=False)
    print(f"数据合并完成，已保存至 {output_path}")
    return merged_df


if __name__ == "__main__":
    # 数据文件路径
    cutter_layout_path = "./data/Cutter_layout_template.xlsx"
    tunnel_data_path = "./data/tunnel_data.xlsx"
    wear_data_path = "./data/wear_data_new.xlsx"
    output_path = "./data/merged_tunnel_data_new.xlsx"

    # 执行数据合并
    merged_data = merge_tunnel_data(
        cutter_layout_path,
        tunnel_data_path,
        wear_data_path,
        output_path
    )

    # 定义预测相关列
    prediction_columns = [
        col for col in merged_data.columns
        if (col.startswith('wear_') and col.endswith('_increment')) or
           (col.startswith('wear_') and col.endswith('_total')) or
           col == 'energy' or
           col == 'advance_speed_mm_per_min_mean'
    ]
