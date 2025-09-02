# -*- coding: utf-8 -*-
"""
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ======== 可视化中文与负号设置 ========
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ======== 目录准备 ========
for p in ['models', 'results', 'results/prediction_plots', 'results/feature_importance']:
    if not os.path.exists(p):
        os.makedirs(p)


class TBMRegressionModel:
    def __init__(self, data_path, target_columns=None):
        """
        读取数据、确定特征与目标、初始化模型与参数搜索空间。
        """
        # ---- 1) 读取与基础清洗 ----
        self.data = pd.read_excel(data_path)
        self.data = self.data.select_dtypes(include=[np.number])  # 只保留数值列
        self.data = self.data.dropna()  # 删除缺失值

        print(f"数据集形状: {self.data.shape}")
        print(f"数据列名: {list(self.data.columns)}")

        # ---- 2) 目标列定义 ----
        default_targets = ['energy'] + [f'cutter_{i}_wear' for i in range(1, 42)]
        self.target_columns = target_columns if target_columns else default_targets
        # 只保留数据中真实存在的目标列
        self.target_columns = [col for col in self.target_columns if col in self.data.columns]
        print(f"预测目标列({len(self.target_columns)}个): {self.target_columns}")

        # ---- 3) 特征列 = 数值列 - 目标列 - segment_id（如果存在） ----
        exclude_cols = self.target_columns.copy()
        if 'segment_id' in self.data.columns:
            exclude_cols.append('segment_id')

        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        print(f"特征列数量: {len(self.feature_columns)}")
        print(f"特征列: {self.feature_columns}")

        # 当前数据全是数值特征
        self.categorical_features = []
        self.numerical_features = self.feature_columns

        # 分离 X / y
        self.X = self.data[self.feature_columns]
        self.y = self.data[self.target_columns]

        # ---- 4) 候选模型集合 ----
        self.models = {
            '随机森林': RandomForestRegressor(random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            '梯度提升': GradientBoostingRegressor(random_state=42),
            '弹性网络': ElasticNet(random_state=42),
        }

        # ---- 5) 各模型的网格搜索空间 ----
        self.param_grids = {
            '随机森林': {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
            },
            'XGBoost': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 6],
            },
            '梯度提升': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 6],
            },
            '弹性网络': {
                'alpha': [0.1, 1, 10],
                'l1_ratio': [0.2, 0.5, 0.8]
            },
        }

        # 训练过程中的临时/最终容器
        self.best_models = {}  # 每个目标的最佳模型
        self.preprocessors = {}  # 每个目标的预处理对象
        self.evaluation_results = {}  # 全部模型在该目标上的评估详情
        self.predictions = {}  # 测试集预测值

    def init_global_preprocessor(self, test_size=0.2, random_state=42):
        """
        划分训练/测试 + 拟合"全局"预处理器
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        print(f"\n训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")

        # 使用MinMaxScaler
        self.global_preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(), self.numerical_features)
            ])

        # 仅在训练集拟合，避免数据泄露
        self.global_preprocessor.fit(self.X_train)

        # 持久化预处理器
        joblib.dump(self.global_preprocessor, 'models/global_preprocessor.pkl')

    def preprocess_data(self, target, k_best=10):
        """
        针对单个目标列进行预处理
        """
        X_train = self.X_train
        X_test = self.X_test
        y_train_single = self.y_train[target]

        # 1) 全局缩放
        X_train_processed = self.global_preprocessor.transform(X_train)
        X_test_processed = self.global_preprocessor.transform(X_test)

        # 2) 特征选择
        selector = None
        if X_train_processed.shape[1] > k_best:
            # 使用互信息进行特征选择
            selector = SelectKBest(mutual_info_regression, k=k_best)
            X_train_processed = selector.fit_transform(X_train_processed, y_train_single)
            X_test_processed = selector.transform(X_test_processed)
            print(f"\n目标变量 {target} 的特征选择结果（选择了 {k_best} 个特征）")

        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'selector': selector,
        }

    def train_models(self):
        """
        逐目标地进行：预处理 -> GridSearchCV -> 评估 -> 持久化
        """
        for target in tqdm(self.target_columns, desc="处理目标变量"):
            print(f"\n开始处理目标变量: {target}")

            # 预处理
            preprocessed_data = self.preprocess_data(target)

            X_train = preprocessed_data['X_train']
            X_test = preprocessed_data['X_test']
            y_train_single = self.y_train[target]
            y_test_single = self.y_test[target]

            best_score = -np.inf
            best_model = None
            best_model_name = ""
            target_results = {}

            # 遍历候选模型，做网格搜索
            for name, model in self.models.items():
                try:
                    pipeline = Pipeline([('model', model)])

                    # 取该模型对应的网格
                    param_grid = {f'model__{k}': v for k, v in self.param_grids.get(name, {}).items()}

                    # 使用3折交叉验证以减少计算时间
                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grid,
                        cv=3,
                        scoring='r2',
                        n_jobs=-1,
                        verbose=0
                    )

                    grid_search.fit(X_train, y_train_single)
                    y_pred = grid_search.predict(X_test)

                    # 计算各项指标
                    mse = mean_squared_error(y_test_single, y_pred)
                    mae = mean_absolute_error(y_test_single, y_pred)
                    r2 = r2_score(y_test_single, y_pred)

                    target_results[name] = {
                        'mse': mse,
                        'mae': mae,
                        'r2': r2,
                        'best_params': grid_search.best_params_,
                        'model': grid_search.best_estimator_
                    }

                    # 维护全局最佳
                    if r2 > best_score:
                        best_score = r2
                        best_model = grid_search.best_estimator_
                        best_model_name = name
                        self.predictions[target] = {
                            'y_true': y_test_single,
                            'y_pred': y_pred,
                            'model_name': name
                        }

                    print(f"  {name}: R² = {r2:.4f}")

                except Exception as e:
                    print(f"模型 {name} 在处理目标 {target} 时出错: {str(e)}")
                    continue

            # 记录该目标的最佳结果
            self.best_models[target] = {
                'model': best_model,
                'name': best_model_name,
                'r2': best_score
            }

            self.preprocessors[target] = {
                'selector': preprocessed_data['selector'],
            }

            self.evaluation_results[target] = target_results

            # 持久化：最佳模型
            joblib.dump(best_model, f'models/best_model_{target}.pkl')
            if preprocessed_data['selector'] is not None:
                joblib.dump(preprocessed_data['selector'], f'models/selector_{target}.pkl')

            print(f"目标 {target} 最佳模型: {best_model_name}, R²: {best_score:.4f}")

        # 绘制预测对比图
        self.plot_predictions_vs_actual()

    def plot_predictions_vs_actual(self):
        """为每个目标变量绘制预测值与真实值的散点对比图"""
        for target, preds in self.predictions.items():
            y_true = preds['y_true']
            y_pred = preds['y_pred']
            model_name = preds.get('model_name', '最佳模型')

            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.6)

            # 理想参考线 y=x
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--')

            # 计算指标
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)

            plt.title(f'{target} 的预测值与真实值对比\n模型: {model_name}, R²: {r2:.4f}')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'results/prediction_plots/{target}_pred_vs_actual.png')
            plt.close()

    def evaluate_models(self):
        """
        汇总每个目标的最佳模型指标
        """
        summary = []
        for target, results in self.evaluation_results.items():
            if not results:
                continue

            best_model_name = self.best_models[target]['name']
            best_r2 = self.best_models[target]['r2']
            best_mse = results[best_model_name]['mse']
            best_mae = results[best_model_name]['mae']

            summary.append({
                '目标变量': target,
                '最佳模型': best_model_name,
                'R2分数': best_r2,
                'MSE': best_mse,
                'MAE': best_mae
            })

        summary_df = pd.DataFrame(summary)
        print("\n模型评估汇总:")
        print(summary_df)

        # 计算平均指标
        avg_r2 = summary_df['R2分数'].mean()
        avg_mse = summary_df['MSE'].mean()
        avg_mae = summary_df['MAE'].mean()

        print(f"\n平均指标 - R²: {avg_r2:.4f}, MSE: {avg_mse:.4f}, MAE: {avg_mae:.4f}")

        summary_df.to_csv('results/model_evaluation_summary.csv', index=False)

        return summary_df

    def predict(self, new_data_path):
        """
        使用训练好的模型对新数据进行预测
        """
        new_data = pd.read_csv(new_data_path).select_dtypes(include=[np.number]).dropna()

        # 对齐训练时的特征列
        X_new = new_data[self.feature_columns]
        X_new_processed = self.global_preprocessor.transform(X_new)

        predictions = {}
        for target, best_info in self.best_models.items():
            model = best_info['model']
            prep = self.preprocessors[target]

            # 应用特征选择
            if prep['selector'] is not None:
                X_new_selected = prep['selector'].transform(X_new_processed)
            else:
                X_new_selected = X_new_processed

            predictions[target] = model.predict(X_new_selected)

        predictions_df = pd.DataFrame(predictions)

        # 如存在 segment_id，则拼接输出
        if 'segment_id' in new_data.columns:
            result_df = pd.concat([new_data['segment_id'], predictions_df], axis=1)
        else:
            result_df = predictions_df

        result_df.to_csv('results/predictions.csv', index=False)
        print("预测完成，结果已保存到 results/predictions.csv")
        return result_df


if __name__ == "__main__":
    # ======== 主入口 ========
    data_path = "./data/merged_tunnel_data_2.xlsx"

    # 初始化模型
    tbm_model = TBMRegressionModel(data_path)

    # 划分数据 + 拟合全局预处理器
    tbm_model.init_global_preprocessor(test_size=0.2, random_state=42)

    # 训练所有目标的模型
    tbm_model.train_models()

    # 评估汇总
    eval_summary = tbm_model.evaluate_models()

    # 示例：使用模型进行预测
    # predictions = tbm_model.predict("new_data.csv")
