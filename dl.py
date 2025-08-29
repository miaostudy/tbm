import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.base import BaseEstimator, RegressorMixin

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('results'):
    os.makedirs('results')


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=2, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, d_model))

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, output_dim)
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.output_projection(x)
        return x


class TransformerWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, output_dim, d_model=64, nhead=2, num_layers=2,
                 dim_feedforward=128, dropout=0.1, epochs=50, batch_size=32, lr=0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        ).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataset)
            if (epoch + 1) % 10 == 0:
                pass

        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
        return outputs.cpu().numpy()

    def get_params(self, deep=True):
        return {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'd_model': self.d_model,
            'nhead': self.nhead,
            'num_layers': self.num_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self.model = TransformerRegressor(
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        return self


class TBMMultiOutputTransformerModel:
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
        self.input_dim = len(self.feature_columns)
        self.output_dim = len(self.target_columns)

        self.models = {
            'Transformer': TransformerWrapper(
                input_dim=self.input_dim,
                output_dim=self.output_dim
            )
        }

        self.param_grids = {
            'Transformer': {
                'd_model': [32, 64],
                'nhead': [2, 4],
                'num_layers': [1, 2],
                'dim_feedforward': [64, 128],
                'epochs': [50, 100],
                'lr': [0.001, 0.0001]
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

        self.input_dim = len(self.processed_feature_names)
        for name in self.models:
            self.models[name].input_dim = self.input_dim

    def train_models(self):
        preprocessed_data = self.preprocess_data()
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']

        for name, model in tqdm(self.models.items(), desc="训练模型"):
            try:
                pipeline = Pipeline([('model', model)])
                from sklearn.model_selection import GridSearchCV
                grid_search = GridSearchCV(
                    pipeline,
                    self.param_grids.get(name, {}),
                    cv=3,
                    scoring='r2',
                    n_jobs=1
                )
                grid_search.fit(X_train, self.y_train.values)
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

        torch.save(self.best_model, 'models/best_transformer_model.pth')

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
        summary_df.to_csv('results/model_evaluation_summary_3.csv', index=False)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='模型名称', y='平均R2分数', data=summary_df)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('results/model_avg_r2.png')
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
        plt.savefig('results/target_best_r2_3.png')
        plt.close()

        return summary_df

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
    best_model = torch.load('models/best_transformer_model.pth')
    best_model.eval()

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
    tbm_model = TBMMultiOutputTransformerModel(data_path)
    tbm_model.init_preprocessor()
    tbm_model.train_models()
    tbm_model.evaluate_models()
