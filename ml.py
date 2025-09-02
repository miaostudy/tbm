import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from tqdm import tqdm

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/prediction_plots'):
    os.makedirs('results/prediction_plots')


class TBMRegressionModel:
    def __init__(self, data_path, target_columns=None):
        self.data = pd.read_excel(data_path)
        self.data = self.data.select_dtypes(include=[np.number])  # 只保留数值列

        self.data = self.data.dropna()
        print(f"保留数值列并删除缺失值后，数据集形状: {self.data.shape}")

        self.target_columns = target_columns if target_columns else [
                                                                        'energy', 'advance_speed_mm_per_min_mean'
                                                                    ] + [f'wear_{i}_total' for i in range(1, 42)]

        self.target_columns = [col for col in self.target_columns if col in self.data.columns]
        print(f"有效目标列: {self.target_columns}")

        exclude_cols = self.target_columns.copy()
        if 'segment_id' in self.data.columns:
            exclude_cols.append('segment_id')

        self.feature_columns = [col for col in self.data.columns if col not in exclude_cols]
        print(f"初始特征列数量: {len(self.feature_columns)}")
        print(f"初始特征列: {self.feature_columns}")

        self.categorical_features = []
        self.numerical_features = self.feature_columns

        self.X = self.data[self.feature_columns]
        self.y = self.data[self.target_columns]

        self.models = {
            '线性回归': LinearRegression(),
            'Ridge回归': Ridge(random_state=42),
            'Lasso回归': Lasso(random_state=42),
            '弹性网络': ElasticNet(random_state=42),
            '随机森林': RandomForestRegressor(random_state=42, n_jobs=-1),
            '梯度提升': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'K近邻': KNeighborsRegressor(n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
            # 'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        }

        self.param_grids = {
            'Ridge回归': {'alpha': [0.1, 1, 10, 100]},
            'Lasso回归': {'alpha': [0.01, 0.1, 1, 10]},
            '弹性网络': {'alpha': [0.1, 1, 10], 'l1_ratio': [0.2, 0.5, 0.8]},
            '随机森林': {'n_estimators': [50, 100], 'max_depth': [None, 10]},
            '梯度提升': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
            'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]},
            'K近邻': {'n_neighbors': [3, 5, 7]},
            'XGBoost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.8, 1.0]
            }
        }

        self.best_models = {}
        self.preprocessors = {}
        self.evaluation_results = {}
        self.final_train_columns = {}
        self.predictions = {}

    def preprocess_data(self, target, n_components=0.95, k_best=15):
        X_train = self.X_train
        X_test = self.X_test
        y_train_single = self.y_train[target]

        X_train_processed = self.global_preprocessor.transform(X_train)
        X_test_processed = self.global_preprocessor.transform(X_test)

        if X_train_processed.shape[1] > k_best:
            selector = SelectKBest(f_regression, k=k_best)
            X_train_selected = selector.fit_transform(X_train_processed, y_train_single)
            X_test_selected = selector.transform(X_test_processed)

            selected_mask = selector.get_support()
            selected_features = [self.processed_feature_names[i] for i in range(len(self.processed_feature_names))
                                 if selected_mask[i]]

            print(f"\n目标变量 {target} 的特征选择结果（选择了 {len(selected_features)} 个特征）:")
            print(selected_features)
        else:
            selector = None
            X_train_selected = X_train_processed
            X_test_selected = X_test_processed
            selected_features = self.processed_feature_names
            print(f"\n目标变量 {target} 特征数量较少，未进行特征选择:")
            print(selected_features)

        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_selected)
        X_test_pca = pca.transform(X_test_selected)
        # X_train_pca = X_train_selected
        # X_test_pca = X_test_selected
        # 记录最终用于训练的特征数量（PCA后的组件数）
        self.final_train_columns[target] = {
            'selected_features': selected_features,
            'pca_components': X_train_pca.shape[1]
        }

        print(f"目标变量 {target} 经PCA处理后，最终训练特征数量: {X_train_pca.shape[1]}")

        return {
            'X_train': X_train_pca,
            'X_test': X_test_pca,
            'selector': selector,
            'pca': pca,
            'selected_features': selected_features
        }

    def init_global_preprocessor(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        print(f"\n训练集大小: {self.X_train.shape}, 测试集大小: {self.X_test.shape}")
        print(f"训练集特征列名: {list(self.X_train.columns)}")

        self.global_preprocessor = ColumnTransformer(
            transformers=[
                ('num', MinMaxScaler(feature_range=(0, 1)), self.numerical_features)
            ])

        self.global_preprocessor.fit(self.X_train)

        self.processed_feature_names = self.numerical_features
        print(f"预处理后特征总数: {len(self.processed_feature_names)}")

        joblib.dump(self.global_preprocessor, 'models/global_preprocessor.pkl')

    def train_models(self, use_pca=True):
        for target in tqdm(self.target_columns, desc="处理目标变量"):
            preprocessed_data = self.preprocess_data(target)
            X_train = preprocessed_data['X_train']
            X_test = preprocessed_data['X_test']
            y_train_single = self.y_train[target]
            y_test_single = self.y_test[target]

            best_score = -np.inf
            best_model = None
            best_model_name = ""
            target_results = {}
            for name, model in self.models.items():
                try:
                    pipeline = Pipeline([
                        ('model', model)
                    ])

                    grid_search = GridSearchCV(
                        pipeline,
                        {f'model__{k}': v for k, v in self.param_grids.get(name, {}).items()},
                        cv=5,
                        scoring='r2',
                        n_jobs=-1
                    )

                    grid_search.fit(X_train, y_train_single)

                    y_pred = grid_search.predict(X_test)

                    if name == best_model_name or best_model is None:
                        self.predictions[target] = {
                            'y_true': y_test_single,
                            'y_pred': y_pred
                        }

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

                    if r2 > best_score:
                        best_score = r2
                        best_model = grid_search.best_estimator_
                        best_model_name = name
                        self.predictions[target] = {
                            'y_true': y_test_single,
                            'y_pred': y_pred,
                            'model_name': name
                        }
                except Exception as e:
                    print(f"模型 {name} 在处理目标 {target} 时出错: {str(e)}")
                    continue

            self.best_models[target] = {
                'model': best_model,
                'name': best_model_name,
                'r2': best_score
            }

            self.preprocessors[target] = {
                'selector': preprocessed_data['selector'],
                'pca': preprocessed_data['pca'],
                'selected_features': preprocessed_data['selected_features']
            }

            self.evaluation_results[target] = target_results

            joblib.dump(best_model, f'models/best_model_{target}.pkl')
            joblib.dump(preprocessed_data['selector'], f'models/selector_{target}.pkl')
            joblib.dump(preprocessed_data['pca'], f'models/pca_{target}.pkl')

        with open('results/feature_selection_summary.txt', 'w', encoding='utf-8') as f:
            for target, info in self.final_train_columns.items():
                f.write(f"目标变量: {target}\n")
                f.write(f"特征选择后的列名: {info['selected_features']}\n")
                f.write(f"PCA处理后的特征数量: {info['pca_components']}\n\n")
        print("\n特征选择结果已保存至 results/feature_selection_summary.txt")

        self.plot_predictions_vs_actual()

    def plot_predictions_vs_actual(self):
        """绘制每个目标变量的预测值与真实值对比图"""
        for target, preds in self.predictions.items():
            y_true = preds['y_true']
            y_pred = preds['y_pred']
            model_name = preds.get('model_name', '最佳模型')

            plt.figure(figsize=(10, 6))

            plt.scatter(y_true, y_pred, alpha=0.6, label='预测值')

            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线')
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            mae = mean_absolute_error(y_true, y_pred)

            plt.title(f'{target} 的预测值与真实值对比\n'
                      f'模型: {model_name}, R²: {r2:.4f}, MSE: {mse:.4f}')
            plt.xlabel('真实值')
            plt.ylabel('预测值')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'results/prediction_plots/{target}_pred_vs_actual.png')
            plt.close()

    def evaluate_models(self):
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
        print(summary_df)
        summary_df.to_csv('results/model_evaluation_summary.csv', index=False)

        model_avg_scores = {}
        for model_name in self.models.keys():
            scores = []
            for target_results in self.evaluation_results.values():
                if model_name in target_results:
                    scores.append(target_results[model_name]['r2'])
            if scores:
                model_avg_scores[model_name] = np.mean(scores)

        if model_avg_scores:
            sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)
            model_names, avg_r2 = zip(*sorted_models)

            plt.figure(figsize=(12, 6))
            sns.barplot(x=list(model_names), y=list(avg_r2))
            plt.title('不同模型在所有目标变量上的平均R2分数')
            plt.xticks(rotation=45)
            plt.ylabel('平均R2分数')
            plt.tight_layout()
            plt.savefig('results/model_avg_r2.png')
            plt.show()

        return summary_df

    def analyze_feature_importance(self, top_n=10):
        """分析特征重要性"""
        print("\n分析特征重要性...")
        importance_results = {}

        for target, best_info in self.best_models.items():
            model = best_info['model']
            model_name = best_info['name']
            preprocessor = self.preprocessors[target]
            if hasattr(model['model'], 'feature_importances_'):
                importances = model['model'].feature_importances_
                features = preprocessor['selected_features']
                if preprocessor['pca'].n_components_ < len(features):
                    continue
                if len(importances) == len(features):
                    indices = np.argsort(importances)[::-1]
                    top_indices = indices[:top_n]

                    importance_results[target] = {
                        'features': [features[i] for i in top_indices],
                        'importances': [importances[i] for i in top_indices]
                    }
                    plt.figure(figsize=(10, 6))
                    plt.barh(range(top_n), [importances[i] for i in top_indices[::-1]], align='center')
                    plt.yticks(range(top_n), [features[i] for i in top_indices[::-1]])
                    plt.xlabel('特征重要性')
                    plt.title(f'{target} 的前{top_n}个重要特征 ({model_name})')
                    plt.tight_layout()
                    plt.savefig(f'results/feature_importance_{target}.png')
                    plt.close()

        all_importances = {}
        for target, imp_info in importance_results.items():
            for feature, imp in zip(imp_info['features'], imp_info['importances']):
                if feature not in all_importances:
                    all_importances[feature] = []
                all_importances[feature].append(imp)

        if all_importances:
            avg_importances = {f: np.mean(imps) for f, imps in all_importances.items()}
            sorted_avg_imp = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]

            plt.figure(figsize=(10, 6))
            features, imps = zip(*sorted_avg_imp[::-1])
            plt.barh(range(len(features)), imps, align='center')
            plt.yticks(range(len(features)), features)
            plt.xlabel('平均特征重要性')
            plt.title(f'所有目标变量的前{top_n}个重要特征')
            plt.tight_layout()
            plt.savefig('results/overall_feature_importance.png')
            plt.show()

    def predict(self, new_data_path):
        new_data = pd.read_csv(new_data_path)
        # 只保留数值列
        new_data = new_data.select_dtypes(include=[np.number])
        # 删除缺失值
        new_data = new_data.dropna()

        X_new = new_data[self.feature_columns]
        X_new_processed = self.global_preprocessor.transform(X_new)
        predictions = {}

        for target, best_info in self.best_models.items():
            model = best_info['model']
            preprocessor = self.preprocessors[target]
            if preprocessor['selector']:
                X_new_selected = preprocessor['selector'].transform(X_new_processed)
            else:
                X_new_selected = X_new_processed
            X_new_pca = preprocessor['pca'].transform(X_new_selected)
            predictions[target] = model.predict(X_new_pca)

        predictions_df = pd.DataFrame(predictions)
        if 'segment_id' in new_data.columns:
            result_df = pd.concat([new_data['segment_id'], predictions_df], axis=1)
        else:
            result_df = predictions_df

        result_df.to_csv('results/predictions.csv', index=False)
        print("预测完成，结果已保存到 results/predictions.csv")

        return result_df


def load_and_predict(new_data_path):
    global_preprocessor = joblib.load('models/global_preprocessor.pkl')
    new_data = pd.read_csv(new_data_path)
    new_data = new_data.select_dtypes(include=[np.number]).dropna()

    target_columns = [f for f in os.listdir('models') if f.startswith('best_model_')]
    target_columns = [f[len('best_model_'):-4] for f in target_columns]

    feature_columns = [col for col in new_data.columns
                       if col not in target_columns and col != 'segment_id']
    X_new = new_data[feature_columns]
    X_new_processed = global_preprocessor.transform(X_new)

    predictions = {}
    for target in target_columns:
        model = joblib.load(f'models/best_model_{target}.pkl')
        try:
            selector = joblib.load(f'models/selector_{target}.pkl')
        except:
            selector = None
        pca = joblib.load(f'models/pca_{target}.pkl')

        if selector:
            X_new_selected = selector.transform(X_new_processed)
        else:
            X_new_selected = X_new_processed

        X_new_pca = pca.transform(X_new_selected)
        predictions[target] = model.predict(X_new_pca)

    predictions_df = pd.DataFrame(predictions)
    if 'segment_id' in new_data.columns:
        result_df = pd.concat([new_data['segment_id'], predictions_df], axis=1)
    else:
        result_df = predictions_df

    return result_df


if __name__ == "__main__":
    data_path = "./data/merged_tunnel_data_new.xlsx"
    tbm_model = TBMRegressionModel(data_path)
    tbm_model.init_global_preprocessor()
    tbm_model.train_models(use_pca=True)
    eval_summary = tbm_model.evaluate_models()
    tbm_model.analyze_feature_importance(top_n=10)
