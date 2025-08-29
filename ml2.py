import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
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


class TBMMultiOutputRegressionModel:
    def __init__(self, data_path, target_columns=None):
        self.data = pd.read_excel(data_path).fillna(0)
        self.target_columns = target_columns if target_columns else [
                                                                        'energy', 'advance_speed_mm_per_min_mean'
                                                                    ] + [f'cutter_{i}_wear' for i in range(1, 42)]

        self.target_columns = [col for col in self.target_columns if col in self.data.columns]
        self.feature_columns = [col for col in self.data.columns
                                if col not in self.target_columns and col != 'segment_id']

        self.categorical_features = []
        self.numerical_features = []

        for col in self.feature_columns:
            if self.data[col].dtype == 'object' or self.data[col].apply(lambda x: isinstance(x, str)).any():
                self.categorical_features.append(col)
            else:
                self.numerical_features.append(col)

        self.X = self.data[self.feature_columns]
        self.y = self.data[self.target_columns]

        self.models = {
            '线性回归': MultiOutputRegressor(LinearRegression()),
            'Ridge回归': MultiOutputRegressor(Ridge(random_state=42)),
            'K近邻': MultiOutputRegressor(KNeighborsRegressor(n_jobs=-1)),
            '随机森林': MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1)),
            '梯度提升': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
            'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)),
            'LightGBM': MultiOutputRegressor(lgb.LGBMRegressor(random_state=42, n_jobs=-1))
        }

        self.param_grids = {
            'Ridge回归': {'estimator__alpha': [0.1, 1, 10, 100]},
            'K近邻': {'estimator__n_neighbors': [3, 5, 7]},
            '随机森林': {
                'estimator__n_estimators': [50, 100],
                'estimator__max_depth': [None, 10]
            },
            '梯度提升': {
                'estimator__n_estimators': [50, 100],
                'estimator__learning_rate': [0.01, 0.1],
                'estimator__max_depth': [3, 5]
            },
            'XGBoost': {
                'estimator__n_estimators': [50, 100],
                'estimator__learning_rate': [0.01, 0.1],
                'estimator__max_depth': [3, 5],
                'estimator__subsample': [0.8, 1.0]
            },
            'LightGBM': {
                'estimator__n_estimators': [50, 100],
                'estimator__learning_rate': [0.01, 0.1],
                'estimator__num_leaves': [31, 63],
                'estimator__subsample': [0.8, 1.0]
            }
        }

        self.best_model = None
        self.best_model_name = ""
        self.best_model_score = -np.inf
        self.preprocessor = None
        self.evaluation_results = {}
        self.processed_feature_names = []

    def preprocess_data(self):
        X_train_processed = self.preprocessor.transform(self.X_train)
        X_test_processed = self.preprocessor.transform(self.X_test)
        return {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'selected_features': self.processed_feature_names
        }

    def init_preprocessor(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), self.numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.categorical_features)
            ])

        self.preprocessor.fit(self.X_train)

        num_feature_names = self.numerical_features
        if self.categorical_features:
            cat_encoder = self.preprocessor.named_transformers_['cat']
            cat_feature_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        else:
            cat_feature_names = []
        self.processed_feature_names = num_feature_names + cat_feature_names
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')

    def train_models(self):
        preprocessed_data = self.preprocess_data()
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']

        for name, model in tqdm(self.models.items(), desc="训练模型"):
            try:
                pipeline = Pipeline([('model', model)])
                grid_search = GridSearchCV(
                    pipeline,
                    self.param_grids.get(name, {}),
                    cv=5,
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train, self.y_train)
                y_pred = grid_search.predict(X_test)

                target_results = {}
                overall_metrics = {'mse': [], 'mae': [], 'r2': []}

                for i, target in enumerate(self.target_columns):
                    y_test_single = self.y_test.iloc[:, i]
                    y_pred_single = y_pred[:, i]

                    mse = mean_squared_error(y_test_single, y_pred_single)
                    mae = mean_absolute_error(y_test_single, y_pred_single)
                    r2 = r2_score(y_test_single, y_pred_single)

                    target_results[target] = {'mse': mse, 'mae': mae, 'r2': r2}
                    overall_metrics['mse'].append(mse)
                    overall_metrics['mae'].append(mae)
                    overall_metrics['r2'].append(r2)

                avg_r2 = np.mean(overall_metrics['r2'])
                avg_mse = np.mean(overall_metrics['mse'])
                avg_mae = np.mean(overall_metrics['mae'])

                self.evaluation_results[name] = {
                    'target_results': target_results,
                    'average_metrics': {'mse': avg_mse, 'mae': avg_mae, 'r2': avg_r2},
                    'best_params': grid_search.best_params_,
                    'model': grid_search.best_estimator_
                }

                if avg_r2 > self.best_model_score:
                    self.best_model_score = avg_r2
                    self.best_model = grid_search.best_estimator_
                    self.best_model_name = name
            except Exception as e:
                continue

        joblib.dump(self.best_model, 'models/best_multioutput_model.pkl')

    def evaluate_models(self):
        summary = []
        for model_name, results in self.evaluation_results.items():
            summary.append({
                '模型名称': model_name,
                '平均R2分数': results['average_metrics']['r2'],
                '平均MSE': results['average_metrics']['mse'],
                '平均MAE': results['average_metrics']['mae']
            })

        summary_df = pd.DataFrame(summary).sort_values(by='平均R2分数', ascending=False)
        summary_df.to_csv('results/model_evaluation_summary_2.csv', index=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='模型名称', y='平均R2分数', data=summary_df)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/model_avg_r2_2.png')
        plt.close()

        target_best_r2 = {}
        for target in self.target_columns:
            best_r2 = -np.inf
            best_model = ""
            for model_name, results in self.evaluation_results.items():
                if results['target_results'][target]['r2'] > best_r2:
                    best_r2 = results['target_results'][target]['r2']
                    best_model = model_name
            target_best_r2[target] = {'best_r2': best_r2, 'best_model': best_model}

        target_df = pd.DataFrame.from_dict(target_best_r2, orient='index').reset_index()
        target_df.columns = ['目标变量', '最佳R2分数', '最佳模型']
        target_df = target_df.sort_values(by='最佳R2分数', ascending=False)

        plt.figure(figsize=(14, 8))
        sns.barplot(x='目标变量', y='最佳R2分数', data=target_df)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('results/target_best_r2_2.png')
        plt.close()

        return summary_df

    def analyze_feature_importance(self, top_n=10):
        if hasattr(self.best_model['model'], 'estimator'):
            base_estimator = self.best_model['model'].estimator
        else:
            base_estimator = self.best_model['model']

        if hasattr(base_estimator, 'feature_importances_'):
            importances = base_estimator.feature_importances_
            features = self.processed_feature_names

            if len(importances) == len(features):
                indices = np.argsort(importances)[::-1]
                top_indices = indices[:top_n]

                plt.figure(figsize=(10, 6))
                plt.barh(range(top_n), [importances[i] for i in top_indices[::-1]], align='center')
                plt.yticks(range(top_n), [features[i] for i in top_indices[::-1]])
                plt.xlabel('特征重要性')
                plt.tight_layout()
                plt.savefig(f'results/feature_importance.png')
                plt.close()

    def predict(self, new_data_path):
        new_data = pd.read_csv(new_data_path)
        X_new = new_data[self.feature_columns]
        X_new_processed = self.preprocessor.transform(X_new)
        predictions = self.best_model.predict(X_new_processed)
        predictions_df = pd.DataFrame(predictions, columns=self.target_columns)
        result_df = pd.concat([new_data['segment_id'], predictions_df], axis=1)
        result_df.to_csv('results/predictions.csv', index=False)
        return result_df


def load_and_predict(new_data_path):
    preprocessor = joblib.load('models/preprocessor.pkl')
    best_model = joblib.load('models/best_multioutput_model.pkl')
    new_data = pd.read_csv(new_data_path)
    target_columns = [col for col in new_data.columns
                      if col not in preprocessor.get_feature_names_out() and col != 'segment_id']
    feature_columns = [col for col in new_data.columns
                       if col in preprocessor.get_feature_names_out() or
                       any(col.startswith(cat) for cat in
                           preprocessor.named_transformers_['cat'].get_feature_names_out())]
    X_new = new_data[feature_columns]
    X_new_processed = preprocessor.transform(X_new)
    predictions = best_model.predict(X_new_processed)
    predictions_df = pd.DataFrame(predictions, columns=target_columns)
    result_df = pd.concat([new_data['segment_id'], predictions_df], axis=1)
    return result_df


if __name__ == "__main__":
    data_path = "./data/merged_tunnel_data.xlsx"
    tbm_model = TBMMultiOutputRegressionModel(data_path)
    tbm_model.init_preprocessor()
    tbm_model.train_models()
    tbm_model.evaluate_models()
    tbm_model.analyze_feature_importance(top_n=10)
