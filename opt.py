import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from scipy.optimize import minimize, basinhopping, differential_evolution
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

# 设置中文字体显示
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 创建优化结果保存目录
if not os.path.exists('optimization_results'):
    os.makedirs('optimization_results')
if not os.path.exists('optimization_results/plots'):
    os.makedirs('optimization_results/plots')


class TBMOptimizer:
    """
    TBM（隧道掘进机）参数优化器
    基于已训练的预测模型，优化掘进参数（推力和转速）以实现多目标优化：
    提高掘进速度、降低能耗和减少刀具磨损
    """

    def __init__(self, model_dir='models', data_path=None):
        """初始化优化器，加载模型和数据"""
        self.model_dir = model_dir
        # 加载全局预处理器
        self.global_preprocessor = joblib.load(os.path.join(model_dir, 'global_preprocessor.pkl'))

        # 识别目标变量（从模型文件名中提取）
        self.target_columns = [f for f in os.listdir(model_dir) if f.startswith('best_model_')]
        self.target_columns = [f[len('best_model_'):-4] for f in self.target_columns]

        # 加载每个目标变量对应的模型和预处理组件
        self.models = {}
        self.preprocessors = {}
        self.expected_feature_counts = {}
        for target in self.target_columns:
            self.models[target] = joblib.load(os.path.join(model_dir, f'best_model_{target}.pkl'))
            selector = joblib.load(os.path.join(model_dir, f'selector_{target}.pkl'))
            pca = joblib.load(os.path.join(model_dir, f'pca_{target}.pkl'))

            self.preprocessors[target] = {
                'selector': selector,  # 特征选择器
                'pca': pca  # PCA降维器
            }

            # 记录每个模型期望的特征数量
            if selector:
                self.expected_feature_counts[target] = selector.n_features_in_
            else:
                self.expected_feature_counts[target] = None

        # 定义可调整的操作参数
        self.adjustable_params = ['main_thrust_kn', 'cutter_head_speed_rpm']
        # 定义目标变量类别
        self.wear_columns = [col for col in self.target_columns if
                             col.startswith('wear_') and col.endswith('_increment')]
        self.energy_column = 'energy'
        self.speed_column = 'advance_speed_mm_per_min_mean'

        # 初始化缺失值填充器
        self.imputer = SimpleImputer(strategy='mean')

        # 初始化数据相关变量
        self.data = None
        self.actual_wear = None
        self.base_feature_names = None
        if data_path:
            self.load_data(data_path)

        # 存储优化结果
        self.optimization_results = {}

    def load_data(self, data_path):
        """加载并预处理数据"""
        self.data = pd.read_excel(data_path)
        # 只保留数值型特征
        self.data = self.data.select_dtypes(include=[np.number])

        # 定义特征列（排除目标变量和ID列）
        self.feature_columns = [col for col in self.data.columns
                                if col not in self.target_columns and col != 'segment_id']

        self.base_feature_names = self.feature_columns.copy()
        # 填充缺失值
        self.data[self.feature_columns] = self.imputer.fit_transform(self.data[self.feature_columns])
        # 移除目标变量或ID列有缺失的行
        self.data = self.data.dropna(subset=self.target_columns + ['segment_id'])
        # 存储每个掘进环的实际磨损数据
        self.actual_wear = {
            seg_id: self.data[self.data['segment_id'] == seg_id][self.wear_columns].iloc[0].to_dict()
            for seg_id in self.data['segment_id'].unique()
        }

    def prepare_features(self, base_features, thrust, speed):
        """
        根据基础特征和调整后的参数（推力和转速）准备输入特征

        参数:
            base_features: 基础特征值
            thrust: 调整后的推力值
            speed: 调整后的转速值

        返回:
            处理后的特征数据，形状为(1, n_features)
        """
        features = base_features.copy()
        # 更新推力和转速相关特征
        for col in features.index:
            if 'main_thrust_kn' in col:
                features[col] = thrust
            elif 'cutter_head_speed_rpm' in col:
                features[col] = speed

        # 确保所有基础特征都存在，缺失的用均值填充
        for feature in self.base_feature_names:
            if feature not in features.index:
                features[feature] = self.imputer.statistics_[self.base_feature_names.index(feature)]

        # 确保特征顺序正确
        features = features.reindex(self.base_feature_names)

        return features.to_frame().T

    def predict(self, features):
        """
        使用加载的模型预测目标变量

        参数:
            features: 输入特征数据

        返回:
            预测结果字典，键为目标变量名，值为预测值
        """
        try:
            # 应用全局预处理
            features_processed = self.global_preprocessor.transform(features)
            # 处理可能的inf和nan值
            features_processed = np.nan_to_num(features_processed)

            predictions = {}
            # 对每个目标变量进行预测
            for target, model in self.models.items():
                preprocessor = self.preprocessors[target]
                selector = preprocessor['selector']
                pca = preprocessor['pca']

                # 处理特征数量不匹配的情况
                if selector:
                    expected = self.expected_feature_counts[target]
                    actual = features_processed.shape[1]

                    if actual != expected:
                        if actual > expected:
                            # 特征过多，截断到期望数量
                            adjusted = features_processed[:, :expected]
                        else:
                            # 特征不足，用0填充
                            adjusted = np.zeros((features_processed.shape[0], expected))
                            adjusted[:, :actual] = features_processed

                        features_selected = selector.transform(adjusted)
                    else:
                        features_selected = selector.transform(features_processed)
                else:
                    features_selected = features_processed

                # 应用PCA降维并预测
                features_pca = pca.transform(features_selected)
                predictions[target] = model.predict(features_pca)[0]

            return predictions
        except Exception as e:
            print(f"预测出错: {str(e)}")
            return None

    def objective_function(self, params, base_features, weights=None):
        """
        目标函数的包装器，支持批量计算

        参数:
            params: 优化参数（推力和转速）
            base_features: 基础特征
            weights: 各目标的权重

        返回:
            目标函数值（单个值或数组）
        """
        params = np.asarray(params)
        if params.ndim > 1:
            # 批量处理参数
            return np.array([self._single_objective(param, base_features, weights) for param in params])
        else:
            # 单个参数处理
            return self._single_objective(params, base_features, weights)

    def _single_objective(self, params, base_features, weights=None):
        """
        单目标函数计算，综合考虑掘进速度、能耗和磨损

        参数:
            params: 优化参数，格式为[thrust, speed]
            base_features: 基础特征
            weights: 各目标的权重，格式为{'speed': ..., 'energy': ..., 'wear': ...}

        返回:
            目标函数值（越小越好）
        """
        thrust, speed = params
        # 准备特征
        features = self.prepare_features(base_features, thrust, speed)
        # 预测目标变量
        predictions = self.predict(features)
        if predictions is None:
            return np.inf  # 预测失败时返回无穷大

        # 设置默认权重
        if weights is None:
            weights = {
                'speed': 1.0,  # 掘进速度权重
                'energy': 1.0,  # 能耗权重
                'wear': 1.0  # 磨损权重
            }

        # 提取预测值
        speed_value = predictions.get(self.speed_column, 0)
        energy_value = predictions.get(self.energy_column, 0)
        wear_values = [predictions[col] for col in self.wear_columns if col in predictions]
        avg_wear = np.mean(wear_values) if wear_values else 0

        # 处理无效的掘进速度
        if speed_value <= 0:
            return np.inf  # 掘进速度不能为零或负值

        # 归一化各目标值（根据领域知识设定合理范围）
        speed_norm = np.clip(speed_value / 100, 0, 1)  # 掘进速度归一化
        energy_norm = np.clip(energy_value / 10, 0, 1)  # 能耗归一化
        wear_norm = np.clip(avg_wear / 0.5, 0, 1)  # 磨损归一化

        # 计算目标函数：速度越大越好（1-speed_norm），能耗和磨损越小越好
        objective = (1 - speed_norm) * weights['speed'] + \
                    energy_norm * weights['energy'] + \
                    wear_norm * weights['wear']
        return float(objective)

    def optimize_segment(self, segment_id, bounds=None, weights=None, method='basinhopping'):
        """
        优化单个掘进环的操作参数

        参数:
            segment_id: 掘进环ID
            bounds: 参数优化范围，格式为[(thrust_min, thrust_max), (speed_min, speed_max)]
            weights: 各目标的权重
            method: 优化方法，可选 'basinhopping', 'differential_evolution' 或其他scipy优化方法

        返回:
            优化结果字典，包含最优参数、预测结果等
        """
        # 获取该掘进环的基础数据
        segment_data = self.data[self.data['segment_id'] == segment_id].iloc[0]
        base_features = segment_data[self.feature_columns]

        # 设置默认参数范围 parameter
        if bounds is None:
            thrust_min = max(500, segment_data.get('main_thrust_kn_mean', 1000) * 0.5)
            thrust_max = min(5000, segment_data.get('main_thrust_kn_mean', 2000) * 1.5)
            speed_min = max(0.5, segment_data.get('cutter_head_speed_rpm_mean', 2) * 0.5)
            speed_max = min(10, segment_data.get('cutter_head_speed_rpm_mean', 5) * 1.5)
            bounds = [(thrust_min, thrust_max), (speed_min, speed_max)]

        # 选择优化方法并执行优化
        if method == 'differential_evolution':
            # 差分进化算法，全局优化 parameter
            result = differential_evolution(
                self.objective_function,
                bounds,
                args=(base_features, weights),
                popsize=15,  # 种群大小
                maxiter=100,  # 最大迭代次数
                tol=0.01,  # 容忍误差
                mutation=(0.5, 1),  # 变异系数
                recombination=0.7,  # 重组概率
                seed=42,  # 随机种子
                workers=1  # 单线程
            )
        elif method == 'basinhopping':
            # 盆地跳跃算法，全局优化
            result = basinhopping(
                self._single_objective,
                x0=[np.mean(bounds[0]), np.mean(bounds[1])],  # 初始点
                minimizer_kwargs={'args': (base_features, weights), 'bounds': bounds},
                niter=30,  # 迭代次数
                seed=42  # 随机种子
            )
            # 确保结果是标量
            result.fun = result.fun if not isinstance(result.fun, np.ndarray) else result.fun.item()
        else:
            # 其他局部优化方法 parameter
            result = minimize(
                self._single_objective,
                x0=[np.mean(bounds[0]), np.mean(bounds[1])],  # 初始点
                args=(base_features, weights),
                method=method,
                bounds=bounds
            )

        # 获取最优参数并预测结果
        optimal_thrust, optimal_speed = result.x
        optimal_features = self.prepare_features(base_features, optimal_thrust, optimal_speed)
        optimal_predictions = self.predict(optimal_features)
        if optimal_predictions is None:
            print(f"掘进环 {segment_id} 优化后预测失败")
            return None

        # 获取原始参数的预测结果作为对比
        original_thrust = segment_data.get('main_thrust_kn_mean', np.mean(bounds[0]))
        original_speed = segment_data.get('cutter_head_speed_rpm_mean', np.mean(bounds[1]))
        original_features = self.prepare_features(base_features, original_thrust, original_speed)
        original_predictions = self.predict(original_features)
        if original_predictions is None:
            print(f"掘进环 {segment_id} 原始参数预测失败")
            return None

        # 保存优化结果
        self.optimization_results[segment_id] = {
            'optimal_params': {
                'thrust': optimal_thrust,
                'speed': optimal_speed
            },
            'original_params': {
                'thrust': original_thrust,
                'speed': original_speed
            },
            'optimal_predictions': optimal_predictions,
            'original_predictions': original_predictions,
            'actual_wear': self.actual_wear.get(segment_id, None),
            'objective_value': result.fun,
            'success': result.success
        }

        return self.optimization_results[segment_id]
    # parameter
    def optimize_all_segments(self, bounds=None, weights=None, method='basinhopping'):
        """
        优化所有掘进环的操作参数

        参数:
            bounds: 参数优化范围
            weights: 各目标的权重
            method: 优化方法

        返回:
            所有掘进环的优化结果
        """
        segment_ids = self.data['segment_id'].unique()
        for seg_id in tqdm(segment_ids, desc="优化所有掘进环"):
            try:
                self.optimize_segment(seg_id, bounds, weights, method)
            except Exception as e:
                print(f"优化掘进环 {seg_id} 时出错: {str(e)}")
                try:
                    # 尝试使用替代优化方法
                    alt_method = 'Nelder-Mead' if method != 'Nelder-Mead' else 'L-BFGS-B'
                    self.optimize_segment(seg_id, bounds, weights, method=alt_method)
                except Exception as e2:
                    print(f"重试优化掘进环 {seg_id} 仍失败: {str(e2)}")

        # 保存结果并生成分析报告
        self.save_results()
        self.generate_analysis_report()

        return self.optimization_results

    def save_results(self):
        """保存优化结果到CSV文件"""
        results_list = []
        for seg_id, result in self.optimization_results.items():
            # 基础信息
            base_info = {
                'segment_id': seg_id,
                'optimal_thrust': result['optimal_params']['thrust'],
                'optimal_speed': result['optimal_params']['speed'],
                'original_thrust': result['original_params']['thrust'],
                'original_speed': result['original_params']['speed'],
                'objective_value': result['objective_value'],
                'success': result['success']
            }
            # 优化后的预测结果
            opt_preds = {
                f'optimal_{key}': value
                for key, value in result['optimal_predictions'].items()
            }
            # 原始参数的预测结果
            orig_preds = {
                f'original_{key}': value
                for key, value in result['original_predictions'].items()
            }
            # 实际磨损数据
            actual_wear = {
                f'actual_{key}': value
                for key, value in result['actual_wear'].items()
            } if result['actual_wear'] else {}

            # 合并所有信息
            results_list.append({**base_info, **opt_preds, **orig_preds, **actual_wear})

        # 保存为CSV
        results_df = pd.DataFrame(results_list)
        results_df.to_csv('optimization_results/optimization_results.csv', index=False)
        print("优化结果已保存至 optimization_results/optimization_results.csv")
        return results_df

    def generate_analysis_report(self):
        """生成优化结果的分析报告和可视化图表"""
        results_df = self.save_results()
        self.plot_optimization_overview(results_df)
        self.plot_parameter_changes(results_df)
        self.plot_objective_improvement(results_df)
        self.plot_wear_comparison(results_df)
        self.generate_numeric_summary(results_df)

    def plot_optimization_overview(self, results_df):
        """绘制优化效果概览图，包括速度、能耗和磨损的改进百分比分布"""
        # 计算改进百分比
        results_df['speed_improvement_pct'] = (
                (results_df[f'optimal_{self.speed_column}'] - results_df[f'original_{self.speed_column}']) /
                results_df[f'original_{self.speed_column}'] * 100
        ).clip(-100, None)  # 限制最小改进为-100%

        results_df['energy_improvement_pct'] = (
                (results_df[f'original_{self.energy_column}'] - results_df[f'optimal_{self.energy_column}']) /
                results_df[f'original_{self.energy_column}'] * 100
        ).clip(-100, None)

        # 计算平均磨损改进百分比
        wear_improvements = []
        for i, row in results_df.iterrows():
            opt_wears = [row[f'optimal_{col}'] for col in self.wear_columns if f'optimal_{col}' in row]
            orig_wears = [row[f'original_{col}'] for col in self.wear_columns if f'original_{col}' in row]

            if opt_wears and orig_wears:
                avg_opt = np.mean(opt_wears)
                avg_orig = np.mean(orig_wears)
                wear_improvements.append((avg_orig - avg_opt) / avg_orig * 100 if avg_orig != 0 else 0)
            else:
                wear_improvements.append(0)

        results_df['wear_improvement_pct'] = wear_improvements

        # 绘制三个改进百分比的分布直方图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        sns.histplot(data=results_df, x='speed_improvement_pct', ax=axes[0], kde=True)
        axes[0].axvline(x=0, color='r', linestyle='--')  # 零改进参考线
        axes[0].set_title(f'掘进速度改进百分比分布\n平均: {results_df["speed_improvement_pct"].mean():.2f}%')
        axes[0].set_xlabel('改进百分比 (%)')

        sns.histplot(data=results_df, x='energy_improvement_pct', ax=axes[1], kde=True)
        axes[1].axvline(x=0, color='r', linestyle='--')
        axes[1].set_title(f'比能改进百分比分布\n平均: {results_df["energy_improvement_pct"].mean():.2f}%')
        axes[1].set_xlabel('改进百分比 (%)')

        sns.histplot(data=results_df, x='wear_improvement_pct', ax=axes[2], kde=True)
        axes[2].axvline(x=0, color='r', linestyle='--')
        axes[2].set_title(f'平均磨损改进百分比分布\n平均: {results_df["wear_improvement_pct"].mean():.2f}%')
        axes[2].set_xlabel('改进百分比 (%)')

        plt.tight_layout()
        plt.savefig('optimization_results/plots/improvement_distributions.png')
        plt.close()

    def plot_parameter_changes(self, results_df):
        """绘制优化前后参数变化和参数相关性图"""
        # 绘制推力和转速优化前后对比散点图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 推力对比
        axes[0].scatter(results_df['original_thrust'], results_df['optimal_thrust'], alpha=0.6)
        max_val = max(results_df['original_thrust'].max(), results_df['optimal_thrust'].max()) * 1.1
        axes[0].plot([0, max_val], [0, max_val], 'r--')  # 对角线参考线
        axes[0].set_title('优化前后的推力对比')
        axes[0].set_xlabel('原始推力 (kN)')
        axes[0].set_ylabel('优化后推力 (kN)')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # 转速对比
        axes[1].scatter(results_df['original_speed'], results_df['optimal_speed'], alpha=0.6)
        max_val = max(results_df['original_speed'].max(), results_df['optimal_speed'].max()) * 1.1
        axes[1].plot([0, max_val], [0, max_val], 'r--')  # 对角线参考线
        axes[1].set_title('优化前后的转速对比')
        axes[1].set_xlabel('原始转速 (RPM)')
        axes[1].set_ylabel('优化后转速 (RPM)')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('optimization_results/plots/parameter_changes.png')
        plt.close()

        # 绘制参数与目标值的相关性热图
        fig, ax = plt.subplots(figsize=(10, 8))
        param_data = results_df[['optimal_thrust', 'optimal_speed',
                                 f'optimal_{self.speed_column}',
                                 f'optimal_{self.energy_column}']]
        param_data.columns = ['推力', '转速', '掘进速度', '比能']

        corr_matrix = param_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('优化参数与目标值的相关性')
        plt.tight_layout()
        plt.savefig('optimization_results/plots/parameter_correlation.png')
        plt.close()

    def plot_objective_improvement(self, results_df):
        """绘制目标函数改进情况对比图"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 掘进速度对比
        axes[0].scatter(results_df[f'original_{self.speed_column}'], results_df[f'optimal_{self.speed_column}'],
                        alpha=0.6)
        max_val = max(results_df[f'original_{self.speed_column}'].max(),
                      results_df[f'optimal_{self.speed_column}'].max()) * 1.1
        axes[0].plot([0, max_val], [0, max_val], 'r--')  # 对角线参考线
        axes[0].set_title('优化前后的掘进速度对比')
        axes[0].set_xlabel('原始掘进速度')
        axes[0].set_ylabel('优化后掘进速度')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # 比能对比
        axes[1].scatter(results_df[f'original_{self.energy_column}'], results_df[f'optimal_{self.energy_column}'],
                        alpha=0.6)
        max_val = max(results_df[f'original_{self.energy_column}'].max(),
                      results_df[f'optimal_{self.energy_column}'].max()) * 1.1
        axes[1].plot([0, max_val], [0, max_val], 'r--')  # 对角线参考线
        axes[1].set_title('优化前后的比能对比')
        axes[1].set_xlabel('原始比能')
        axes[1].set_ylabel('优化后比能')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        # 目标函数值分布
        sns.histplot(data=results_df, x='objective_value', ax=axes[2], kde=True)
        axes[2].set_title('优化目标函数值分布')
        axes[2].set_xlabel('目标函数值 (越小越好)')
        axes[2].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('optimization_results/plots/objective_improvement.png')
        plt.close()

    def plot_wear_comparison(self, results_df):
        """绘制磨损对比图"""
        # 计算平均磨损值
        results_df['avg_original_wear'] = results_df[[f'original_{col}' for col in self.wear_columns
                                                      if f'original_{col}' in results_df.columns]].mean(axis=1)
        results_df['avg_optimal_wear'] = results_df[[f'optimal_{col}' for col in self.wear_columns
                                                     if f'optimal_{col}' in results_df.columns]].mean(axis=1)
        results_df['avg_actual_wear'] = results_df[[f'actual_{col}' for col in self.wear_columns
                                                    if f'actual_{col}' in results_df.columns]].mean(axis=1)

        # 绘制前20环的磨损对比柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_data = results_df.head(20)  # 只显示前20环

        x = np.arange(len(plot_data))
        width = 0.25

        ax.bar(x - width, plot_data['avg_original_wear'], width, label='原始参数预测磨损')
        ax.bar(x, plot_data['avg_optimal_wear'], width, label='优化参数预测磨损')
        ax.bar(x + width, plot_data['avg_actual_wear'], width, label='实际磨损')

        ax.set_xlabel('掘进环ID')
        ax.set_ylabel('平均磨损增量')
        ax.set_title('不同参数下的平均磨损增量对比 (前20环)')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['segment_id'], rotation=45)
        ax.legend()

        plt.tight_layout()
        plt.savefig('optimization_results/plots/wear_comparison.png')
        plt.close()

        # 绘制预测磨损与实际磨损的对比散点图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 原始参数预测 vs 实际
        axes[0].scatter(results_df['avg_actual_wear'], results_df['avg_original_wear'], alpha=0.6)
        max_val = max(results_df['avg_actual_wear'].max(), results_df['avg_original_wear'].max()) * 1.1
        axes[0].plot([0, max_val], [0, max_val], 'r--')  # 对角线参考线
        axes[0].set_title('原始参数预测磨损 vs 实际磨损')
        axes[0].set_xlabel('实际平均磨损')
        axes[0].set_ylabel('原始参数预测平均磨损')
        axes[0].grid(True, linestyle='--', alpha=0.7)

        # 优化参数预测 vs 实际
        axes[1].scatter(results_df['avg_actual_wear'], results_df['avg_optimal_wear'], alpha=0.6)
        max_val = max(results_df['avg_actual_wear'].max(), results_df['avg_optimal_wear'].max()) * 1.1
        axes[1].plot([0, max_val], [0, max_val], 'r--')  # 对角线参考线
        axes[1].set_title('优化参数预测磨损 vs 实际磨损')
        axes[1].set_xlabel('实际平均磨损')
        axes[1].set_ylabel('优化参数预测平均磨损')
        axes[1].grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.savefig('optimization_results/plots/wear_vs_actual.png')
        plt.close()

    def generate_numeric_summary(self, results_df):
        """生成优化结果的数值摘要报告"""
        summary = []
        # 计算掘进速度改进
        speed_improvement = (
                (results_df[f'optimal_{self.speed_column}'].mean() - results_df[
                    f'original_{self.speed_column}'].mean()) /
                results_df[f'original_{self.speed_column}'].mean() * 100
        )

        # 计算比能改进
        energy_improvement = (
                (results_df[f'original_{self.energy_column}'].mean() - results_df[
                    f'optimal_{self.energy_column}'].mean()) /
                results_df[f'original_{self.energy_column}'].mean() * 100
        )

        # 计算磨损改进
        orig_avg_wears = results_df[[f'original_{col}' for col in self.wear_columns
                                     if f'original_{col}' in results_df.columns]].mean()
        opt_avg_wears = results_df[[f'optimal_{col}' for col in self.wear_columns
                                    if f'optimal_{col}' in results_df.columns]].mean()
        wear_improvement = ((orig_avg_wears - opt_avg_wears) / orig_avg_wears * 100).mean()

        # 添加摘要信息
        summary.append({
            '指标': '平均掘进速度改进',
            '原始平均值': results_df[f'original_{self.speed_column}'].mean(),
            '优化后平均值': results_df[f'optimal_{self.speed_column}'].mean(),
            '改进百分比(%)': speed_improvement
        })

        summary.append({
            '指标': '平均比能改进',
            '原始平均值': results_df[f'original_{self.energy_column}'].mean(),
            '优化后平均值': results_df[f'optimal_{self.energy_column}'].mean(),
            '改进百分比(%)': energy_improvement
        })

        summary.append({
            '指标': '平均磨损改进',
            '原始平均值': orig_avg_wears.mean(),
            '优化后平均值': opt_avg_wears.mean(),
            '改进百分比(%)': wear_improvement
        })

        # 为每个磨损列添加详细信息
        for col in self.wear_columns:
            if f'original_{col}' in results_df.columns and f'optimal_{col}' in results_df.columns:
                orig_mean = results_df[f'original_{col}'].mean()
                opt_mean = results_df[f'optimal_{col}'].mean()
                improvement = (orig_mean - opt_mean) / orig_mean * 100 if orig_mean != 0 else 0

                summary.append({
                    '指标': f'{col} 改进',
                    '原始平均值': orig_mean,
                    '优化后平均值': opt_mean,
                    '改进百分比(%)': improvement
                })

        # 保存摘要报告
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('optimization_results/optimization_summary.csv', index=False)
        print("优化摘要已保存至 optimization_results/optimization_summary.csv")

        return summary_df


if __name__ == "__main__":
    # 初始化优化器
    optimizer = TBMOptimizer(
        model_dir='models',
        data_path="./data/merged_tunnel_data_new.xlsx"
    )

    # 设置各目标的权重 parameter
    weights = {
        'speed': 1.2,  # 掘进速度权重（越大越优先）
        'energy': 1.0,  # 比能权重
        'wear': 0.8  # 磨损量权重
    }

    print("开始优化所有掘进环的参数...")
    results = optimizer.optimize_all_segments(
        method='basinhopping',  # 使用盆地跳跃算法进行全局优化
        weights=weights
    )
    print("优化完成！结果已保存至 optimization_results 目录")
