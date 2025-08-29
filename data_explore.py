import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from math import ceil

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def categorize_features(df):
    target_vars = ['advance_speed_mm_per_min_mean', 'energy'] + \
                  [f'cutter_{i}_wear' for i in range(1, 42)]

    existing_targets = [var for var in target_vars if var in df.columns]
    cutter_attrs = []
    for i in range(1, 42):
        cutter_attrs.extend([
            f'cutter_{i}_radius_mm',
            f'cutter_{i}_type',
            f'cutter_{i}_installation_plane_angle',
            f'cutter_{i}_inclination_to_axis',
            f'cutter_{i}_wear_limit_mm',
            f'cutter_{i}_neighbors'
        ])
    cutter_attrs = [attr for attr in cutter_attrs if attr in df.columns]

    geo_machine_params = ['ucs_mpa', 'cai', 'bts_mpa', 'des_g_per_mm2', 'bq', 'tunneling_distance']
    geo_machine_params = [p for p in geo_machine_params if p in df.columns]

    operation_params = []
    param_groups = [
        'cutter_head_speed_rpm', 'main_thrust_kn', 'torque_pressure_bar',
        'cutter_head_torque_kNm', 'auxiliary_gripper_pressure_bar',
        'auxiliary_thrust_cylinder', 'absorption_pump_m', 'oil_humidity_sensor',
        'hydraulic_oil_level', 'lubrication_pressure_ptx97', 'main_thrust_cylinder',
        'main_thrust_piston_rod', 'closed_loop_water_system_status',
        'backup_rotary_conveying_pressure', 'tbm_rotary_conveying_pressure',
        'thrust_proportion_valve', 'return_oil_pressure',
        'rolling_shield_gripper_status', 'stabilizer_pressure_bar',
        'closed_loop_water_temperature', 'torque_motor'
    ]

    for group in param_groups:
        group_params = [col for col in df.columns if group in col]
        operation_params.extend(group_params)

    all_features = set(df.columns)
    categorized = set(existing_targets + cutter_attrs + geo_machine_params + operation_params)
    uncategorized = [col for col in all_features if col not in categorized and col != 'segment_id']

    if uncategorized:
        print(f"注意: 以下特征未被分类: {uncategorized[:5]}...")  # 只显示前5个

    return {
        'targets': existing_targets,
        'cutter_attributes': cutter_attrs,
        'geological_machine': geo_machine_params,
        'operation_parameters': operation_params
    }


def visualize_data(df, feature_categories):
    if not os.path.exists('tbm_visualizations'):
        os.makedirs('tbm_visualizations')

    targets = feature_categories['targets']
    features = df.drop(columns=['segment_id'] + targets if 'segment_id' in df.columns else targets)
    numeric_features = features.select_dtypes(include=['float64', 'int64']).columns

    plot_features_distributions_and_qq(df, numeric_features, "all_features")
    if targets:
        visualize_target_variables(df, targets)
    if targets and not features.empty:
        analyze_feature_target_correlations(df, features, targets)
    plot_correlation_heatmaps(df, features)
    plot_boxplots(df, numeric_features)
    visualize_feature_groups(df, feature_categories)

def plot_features_distributions_and_qq(df, features, prefix, cols=3, rows=4):
    features_per_page = cols * rows * 2
    num_pages = ceil(len(features) / features_per_page)

    for page in range(num_pages):
        start_idx = page * features_per_page // 2
        end_idx = min(start_idx + features_per_page // 2, len(features))
        page_features = features[start_idx:end_idx]

        fig, axes = plt.subplots(rows * 2, cols, figsize=(5 * cols, 4 * rows * 2))
        fig.suptitle(f'特征分布与Q-Q图 (第{page + 1}/{num_pages}页)', y=1.02, fontsize=16)

        for i, feature in enumerate(page_features):
            ax1 = axes[i * 2 // cols, i * 2 % cols]
            sns.histplot(df[feature], kde=True, ax=ax1)
            ax1.set_title(f'{feature} 分布')
            ax1.tick_params(axis='x', rotation=45)

            ax2 = axes[(i * 2 + 1) // cols, (i * 2 + 1) % cols]
            stats.probplot(df[feature], plot=ax2)
            ax2.set_title(f'{feature} Q-Q图')

        plt.tight_layout()
        plt.savefig(f'tbm_visualizations/{prefix}_distribution_qq_page{page + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()


def visualize_target_variables(df, targets):
    cols = 2
    rows = ceil(len(targets) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = axes.flatten() if rows * cols > 1 else [axes]

    for i, target in enumerate(targets):
        sns.histplot(df[target], kde=True, ax=axes[i])
        axes[i].set_title(f'{target} 分布')
        axes[i].tick_params(axis='x', rotation=45)
    for i in range(len(targets), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('tbm_visualizations/target_distributions.png', dpi=300)
    plt.close()

    if len(targets) >= 2:
        plt.figure(figsize=(10, 8))
        corr = df[targets].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title('目标变量间相关性热力图')
        plt.tight_layout()
        plt.savefig('tbm_visualizations/targets_correlation.png', dpi=300)
        plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df[targets])
    plt.xticks(rotation=45)
    plt.title('目标变量箱型图')
    plt.tight_layout()
    plt.savefig('tbm_visualizations/targets_boxplot.png', dpi=300)
    plt.close()


def analyze_feature_target_correlations(df, features, targets):
    numeric_features = features.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_features) == 0:
        return
    combined = pd.concat([df[numeric_features], df[targets]], axis=1)
    corr = combined.corr()
    target_corr = corr[targets].drop(targets, errors='ignore')
    if len(target_corr) > 50:
        abs_corr = target_corr.abs().mean(axis=1).sort_values(ascending=False)
        top_features = abs_corr.head(50).index
        target_corr = target_corr.loc[top_features]

    plt.figure(figsize=(10, len(target_corr) * 0.3))
    sns.heatmap(target_corr, annot=False, cmap='coolwarm', fmt=".2f",
                linewidths=.5, cbar_kws={"shrink": .8})
    plt.title('特征与目标变量相关性热力图')
    plt.tight_layout()
    plt.savefig('tbm_visualizations/features_targets_correlation.png', dpi=300)
    plt.close()

    for target in targets[:5]:
        corr_values = combined.corr()[target].sort_values(ascending=False)
        top_positive = corr_values[1:11]
        top_negative = corr_values[-10:]

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 1, 1)
        sns.barplot(x=top_positive.values, y=top_positive.index)
        plt.title(f'与 {target} 正相关性最高的10个特征')
        plt.subplot(2, 1, 2)
        sns.barplot(x=top_negative.values, y=top_negative.index)
        plt.title(f'与 {target} 负相关性最高的10个特征')

        plt.tight_layout()
        plt.savefig(f'tbm_visualizations/top_correlations_with_{target}.png', dpi=300)
        plt.close()
        top_features = corr_values.drop(target).abs().sort_values(ascending=False).head(6).index
        plt.figure(figsize=(15, 10))

        for i, feature in enumerate(top_features):
            plt.subplot(2, 3, i + 1)
            sns.regplot(x=df[feature], y=df[target], scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
            plt.title(f'{feature} vs {target}\n相关性: {corr_values[feature]:.2f}')

        plt.tight_layout()
        plt.savefig(f'tbm_visualizations/scatter_{target}_vs_top_features.png', dpi=300)
        plt.close()


def plot_correlation_heatmaps(df, features, max_features=50):
    numeric_features = features.select_dtypes(include=['float64', 'int64']).columns

    if len(numeric_features) <= max_features:
        corr = df[numeric_features].corr()
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', fmt=".2f",
                    square=True, linewidths=.5, cbar_kws={"shrink": .8})
        plt.title(f'特征相关性热力图 (共{len(numeric_features)}个特征)')
        plt.tight_layout()
        plt.savefig('tbm_visualizations/features_correlation.png', dpi=300)
        plt.close()
    else:
        num_groups = ceil(len(numeric_features) / max_features)
        for i in range(num_groups):
            start_idx = i * max_features
            end_idx = min(start_idx + max_features, len(numeric_features))
            group_features = numeric_features[start_idx:end_idx]

            corr = df[group_features].corr()
            plt.figure(figsize=(12, 10))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, cmap='coolwarm', fmt=".2f",
                        square=True, linewidths=.5, cbar_kws={"shrink": .8})
            plt.title(f'特征相关性热力图 (第{i + 1}/{num_groups}组, 共{len(group_features)}个特征)')
            plt.tight_layout()
            plt.savefig(f'tbm_visualizations/features_correlation_group{i + 1}.png', dpi=300)
            plt.close()


def plot_boxplots(df, features, cols=3, rows=4):
    features_per_page = cols * rows
    num_pages = ceil(len(features) / features_per_page)

    for page in range(num_pages):
        start_idx = page * features_per_page
        end_idx = min(start_idx + features_per_page, len(features))
        page_features = features[start_idx:end_idx]

        plt.figure(figsize=(5 * cols, 4 * rows))

        for i, feature in enumerate(page_features):
            plt.subplot(rows, cols, i + 1)
            sns.boxplot(y=df[feature])
            plt.title(f'{feature}')

        plt.tight_layout()
        plt.savefig(f'tbm_visualizations/boxplots_page{page + 1}.png', dpi=300)
        plt.close()


def visualize_feature_groups(df, feature_categories):
    if feature_categories['cutter_attributes']:
        cutter_attrs = feature_categories['cutter_attributes']
        numeric_cutter = [col for col in cutter_attrs if df[col].dtype in ['float64', 'int64']]

        if numeric_cutter:
            plot_features_distributions_and_qq(df, numeric_cutter, "cutter_attributes")
            if feature_categories['targets']:
                top_target = feature_categories['targets'][0]
                cutter_corr = df[numeric_cutter + [top_target]].corr()[top_target].sort_values(ascending=False)

                plt.figure(figsize=(10, 8))
                sns.barplot(x=cutter_corr.drop(top_target).values, y=cutter_corr.drop(top_target).index)
                plt.title(f'刀具属性与 {top_target} 的相关性')
                plt.tight_layout()
                plt.savefig('tbm_visualizations/cutter_attrs_vs_target.png', dpi=300)
                plt.close()
    if feature_categories['geological_machine']:
        geo_params = feature_categories['geological_machine']
        if len(geo_params) >= 4:
            sns.pairplot(df[geo_params[:4]], diag_kind='kde')
            plt.suptitle('地质与机器参数两两关系', y=1.02)
            plt.savefig('tbm_visualizations/geo_params_pairplot.png', dpi=300)
            plt.close()
        if feature_categories['targets']:
            targets = feature_categories['targets'][:3]
            for target in targets:
                plt.figure(figsize=(15, 4 * len(geo_params[:5])))

                for i, param in enumerate(geo_params[:5]):
                    plt.subplot(len(geo_params[:5]), 1, i + 1)
                    sns.regplot(x=df[param], y=df[target], scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
                    plt.title(f'{param} vs {target}')

                plt.tight_layout()
                plt.savefig(f'tbm_visualizations/geo_params_vs_{target}.png', dpi=300)
                plt.close()
    if feature_categories['operation_parameters']:
        op_params = feature_categories['operation_parameters']
        numeric_ops = [col for col in op_params if df[col].dtype in ['float64', 'int64']]

        if numeric_ops:
            plot_features_distributions_and_qq(df, numeric_ops[:20], "operation_parameters")
            if len(numeric_ops) >= 2:
                plot_correlation_heatmaps(df, df[numeric_ops[:20]], max_features=20)

def main(file_path):
    df = pd.read_excel(file_path)
    feature_cats = categorize_features(df)
    print(f"目标变量数量: {len(feature_cats['targets'])}")
    print(f"刀具属性特征数量: {len(feature_cats['cutter_attributes'])}")
    print(f"地质与机器参数数量: {len(feature_cats['geological_machine'])}")
    print(f"操作参数数量: {len(feature_cats['operation_parameters'])}")

    visualize_data(df, feature_cats)

if __name__ == "__main__":
    data_file_path = "./data/merged_tunnel_data.xlsx"
    main(data_file_path)
