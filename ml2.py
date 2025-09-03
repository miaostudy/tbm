import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import lightgbm as lgb
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


# Stacking回归特征相关函数
def stacking_reg(clf, train_x, train_y, test_x, clf_name, kf, label_split=None):
    """
    实现stacking回归特征生成

    参数:
        clf: 分类器/回归器
        train_x: 训练特征
        train_y: 训练标签
        test_x: 测试特征
        clf_name: 分类器名称
        kf: KFold对象
        label_split: 用于分层的标签

    返回:
        train: 训练集的stacking特征
        test: 测试集的stacking特征
    """
    train = np.zeros((train_x.shape[0], train_y.shape[1])) if len(train_y.shape) > 1 else np.zeros(
        (train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], train_y.shape[1])) if len(train_y.shape) > 1 else np.zeros((test_x.shape[0], 1))
    folds = kf.get_n_splits(train_x)
    test_pre = np.empty((folds, test_x.shape[0], train_y.shape[1])) if len(train_y.shape) > 1 else np.empty(
        (folds, test_x.shape[0], 1))
    cv_scores = []

    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x, tr_y = train_x[train_index], train_y[train_index]
        te_x, te_y = train_x[test_index], train_y[test_index]

        if clf_name in ["rf", "ada", "gb", "et", "lr", "transformer"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, train_y.shape[1]) if len(train_y.shape) > 1 else clf.predict(
                te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, train_y.shape[1]) if len(
                train_y.shape) > 1 else clf.predict(test_x).reshape(-1, 1)

            # 计算MSE分数
            if len(train_y.shape) > 1:
                # 多输出情况，计算每个输出的MSE再平均
                mse = np.mean([mean_squared_error(te_y[:, j], pre[:, j]) for j in range(train_y.shape[1])])
            else:
                mse = mean_squared_error(te_y, pre)
            cv_scores.append(mse)

        elif clf_name == "xgb":
            # 处理多输出情况，为每个目标创建一个模型
            if len(train_y.shape) > 1 and train_y.shape[1] > 1:
                pre = np.zeros((te_x.shape[0], train_y.shape[1]))
                test_pred = np.zeros((test_x.shape[0], train_y.shape[1]))

                for j in range(train_y.shape[1]):
                    train_matrix = xgb.DMatrix(tr_x, label=tr_y[:, j], missing=-1)
                    test_matrix = xgb.DMatrix(te_x, label=te_y[:, j], missing=-1)
                    z = xgb.DMatrix(test_x, missing=-1)

                    params = {
                        'booster': 'gbtree',
                        'eval_metric': 'rmse',
                        'gamma': 1,
                        'min_child_weight': 1.5,
                        'max_depth': 5,
                        'lambda': 10,
                        'subsample': 0.7,
                        'colsample_bytree': 0.7,
                        'colsample_bylevel': 0.7,
                        'eta': 0.03,
                        'tree_method': 'exact',
                        'seed': 2017,
                        'nthread': 12
                    }

                    watchlist = [(train_matrix, 'train'), (test_matrix, 'eval')]
                    model = xgb.train(
                        params=params,
                        dtrain=train_matrix,
                        num_boost_round=10000,
                        evals=watchlist,
                        early_stopping_rounds=100,
                        verbose_eval=False
                    )

                    pre[:, j] = model.predict(test_matrix)
                    test_pred[:, j] = model.predict(z)

                train[test_index] = pre
                test_pre[i, :] = test_pred
                mse = np.mean([mean_squared_error(te_y[:, j], pre[:, j]) for j in range(train_y.shape[1])])
                cv_scores.append(mse)

            else:
                # 单输出情况
                train_matrix = xgb.DMatrix(tr_x, label=tr_y.ravel(), missing=-1)
                test_matrix = xgb.DMatrix(te_x, label=te_y.ravel(), missing=-1)
                z = xgb.DMatrix(test_x, missing=-1)

                params = {
                    'booster': 'gbtree',
                    'eval_metric': 'rmse',
                    'gamma': 1,
                    'min_child_weight': 1.5,
                    'max_depth': 10,
                    'lambda': 10,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'colsample_bylevel': 0.7,
                    'eta': 0.03,
                    'tree_method': 'exact',
                    'seed': 2017,
                    'nthread': 12
                }

                watchlist = [(train_matrix, 'train'), (test_matrix, 'eval')]
                model = xgb.train(
                    params,
                    train_matrix,
                    num_boost_round=10000,
                    evals=watchlist,
                    early_stopping_rounds=100,
                    verbose_eval=False
                )

                pre = model.predict(test_matrix).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(z).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name == "lgb":
            # 处理多输出情况
            if len(train_y.shape) > 1 and train_y.shape[1] > 1:
                pre = np.zeros((te_x.shape[0], train_y.shape[1]))
                test_pred = np.zeros((test_x.shape[0], train_y.shape[1]))

                for j in range(train_y.shape[1]):
                    train_matrix = lgb.Dataset(tr_x, label=tr_y[:, j])
                    test_matrix = lgb.Dataset(te_x, label=te_y[:, j], reference=train_matrix)

                    params = {
                        'boosting_type': 'gbdt',
                        'objective': 'regression',
                        'metric': 'mse',
                        'min_child_weight': 1.5,
                        'num_leaves': 2 ** 5,
                        'lambda_l2': 10,
                        'subsample': 0.7,
                        'colsample_bytree': 0.7,
                        'colsample_bylevel': 0.7,
                        'learning_rate': 0.03,
                        'seed': 2017,
                        'nthread': 12,
                        'silent': True,
                    }

                    model = lgb.train(
                        params=params,
                        train_set=train_matrix,
                        num_boost_round=10000,
                        valid_sets=test_matrix,
                    )

                    pre[:, j] = model.predict(te_x, num_iteration=model.best_iteration)
                    test_pred[:, j] = model.predict(test_x, num_iteration=model.best_iteration)

                train[test_index] = pre
                test_pre[i, :] = test_pred
                mse = np.mean([mean_squared_error(te_y[:, j], pre[:, j]) for j in range(train_y.shape[1])])
                cv_scores.append(mse)

            else:
                # 单输出情况
                train_matrix = lgb.Dataset(tr_x, label=tr_y.ravel())
                test_matrix = lgb.Dataset(te_x, label=te_y.ravel(), reference=train_matrix)

                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'mse',
                    'min_child_weight': 1.5,
                    'num_leaves': 2 ** 5,
                    'lambda_l2': 10,
                    'subsample': 0.7,
                    'colsample_bytree': 0.7,
                    'colsample_bylevel': 0.7,
                    'learning_rate': 0.03,
                    'seed': 2017,
                    'nthread': 12,
                    'silent': True,
                }

                model = lgb.train(
                    params=params,
                    train_set=train_matrix,
                    num_boost_round=10000,
                    valid_sets=test_matrix,
                )

                pre = model.predict(te_x, num_iteration=model.best_iteration).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))

        else:
            raise ValueError(f"不支持的分类器: {clf_name}")

        print(f"{clf_name} 第{i + 1}折交叉验证分数: {cv_scores[-1]:.4f}")

    test[:] = test_pre.mean(axis=0)
    print(f"{clf_name} 交叉验证分数列表: {[round(score, 4) for score in cv_scores]}")
    print(f"{clf_name} 平均交叉验证分数: {np.mean(cv_scores):.4f}")

    return train, test


# 定义各种基础模型的stacking函数
def rf_reg(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestRegressor(
        n_estimators=300,  # 减少数量以提高速度
        max_depth=15,
        n_jobs=-1,
        random_state=2017,
        verbose=0
    )
    rf_train, rf_test = stacking_reg(randomforest, x_train, y_train, x_valid, "rf", kf, label_split=label_split)
    return rf_train, rf_test, "rf_reg"


def ada_reg(x_train, y_train, x_valid, kf, label_split=None):
    adaboost = AdaBoostRegressor(
        n_estimators=50,
        random_state=2017,
        learning_rate=0.01
    )
    ada_train, ada_test = stacking_reg(adaboost, x_train, y_train, x_valid, "ada", kf, label_split=label_split)
    return ada_train, ada_test, "ada_reg"


def gb_reg(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingRegressor(
        subsample=0.8,
        random_state=2017,
        verbose=0
    )
    gbdt_train, gbdt_test = stacking_reg(gbdt, x_train, y_train, x_valid, "gb", kf, label_split=label_split)
    return gbdt_train, gbdt_test, "gb_reg"


def et_reg(x_train, y_train, x_valid, kf, label_split=None):
    extratree = ExtraTreesRegressor(
        n_jobs=-1,
        random_state=2017,
        verbose=0
    )
    et_train, et_test = stacking_reg(extratree, x_train, y_train, x_valid, "et", kf, label_split=label_split)
    return et_train, et_test, "et_reg"


def lr_reg(x_train, y_train, x_valid, kf, label_split=None):
    lr_reg = LinearRegression(n_jobs=-1)
    lr_train, lr_test = stacking_reg(lr_reg, x_train, y_train, x_valid, "lr", kf, label_split=label_split)
    return lr_train, lr_test, "lr_reg"


def xgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    xgb_train, xgb_test = stacking_reg(None, x_train, y_train, x_valid, "xgb", kf, label_split=label_split)
    return xgb_train, xgb_test, "xgb_reg"


def lgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    lgb_train, lgb_test = stacking_reg(None, x_train, y_train, x_valid, "lgb", kf, label_split=label_split)
    return lgb_train, lgb_test, "lgb_reg"


def transformer_reg(x_train, y_train, x_valid, kf, label_split=None):
    # 根据目标变量数量调整Transformer
    output_dim = y_train.shape[1] if len(y_train.shape) > 1 else 1
    transformer = TransformerWrapper(
        input_dim=x_train.shape[1],
        output_dim=output_dim,
        epochs=30,  # 减少训练轮数以加速stacking过程
        batch_size=32,
        lr=0.001
    )
    tf_train, tf_test = stacking_reg(transformer, x_train, y_train, x_valid, "transformer", kf, label_split=label_split)
    return tf_train, tf_test, "transformer_reg"


# Transformer模型定义
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
        # 处理多输出情况
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

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

        # 初始化模型 - 包括stacking模型和最终模型
        self.stacking_models = [lgb_reg, xgb_reg, rf_reg, et_reg]
        self.models = {
            'Transformer': TransformerWrapper(
                input_dim=self.input_dim,
                output_dim=self.output_dim
            ),
            'Stacking_Transformer': TransformerWrapper(  # 使用stacking特征的Transformer
                input_dim=self.input_dim + len(self.stacking_models) * self.output_dim,
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
            },
            'Stacking_Transformer': {
                'd_model': [64, 128],  # 输入维度增加，适当增大模型容量
                'nhead': [2, 4],
                'num_layers': [2, 3],
                'dim_feedforward': [128, 256],
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
        self.stacking_train = None
        self.stacking_test = None

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
        # 更新模型输入维度
        for name in self.models:
            if name == 'Stacking_Transformer':
                self.models[name].input_dim = self.input_dim + len(self.stacking_models) * self.output_dim
            else:
                self.models[name].input_dim = self.input_dim

    def generate_stacking_features(self):
        """生成stacking特征"""
        preprocessed_data = self.preprocess_data()
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']

        # 处理inf和nan值
        X_train = self.clean_data(X_train)
        X_test = self.clean_data(X_test)

        # 设置stacking的5折交叉验证
        folds = 5
        kf = KFold(n_splits=folds, shuffle=True, random_state=42)

        # 生成stacking特征
        train_data_list = []
        test_data_list = []

        print("开始生成stacking特征...")
        for clf in tqdm(self.stacking_models, desc="训练stacking基础模型"):
            train_data, test_data, clf_name = clf(X_train, self.y_train.values, X_test, kf)
            train_data_list.append(train_data)
            test_data_list.append(test_data)
            print(f"完成 {clf_name} 的stacking特征生成")

        # 拼接所有stacking特征
        self.stacking_train = np.concatenate(train_data_list, axis=1)
        self.stacking_test = np.concatenate(test_data_list, axis=1)

        # 保存stacking特征
        np.save('models/stacking_train.npy', self.stacking_train)
        np.save('models/stacking_test.npy', self.stacking_test)

        print(f"stacking特征生成完成，训练集形状: {self.stacking_train.shape}, 测试集形状: {self.stacking_test.shape}")

    def clean_data(self, data):
        """处理数据中的inf和nan值"""
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data

    def train_models(self):
        preprocessed_data = self.preprocess_data()
        X_train = preprocessed_data['X_train']
        X_test = preprocessed_data['X_test']

        # 如果还没有生成stacking特征，则先生成
        if self.stacking_train is None or self.stacking_test is None:
            self.generate_stacking_features()

        # 为Stacking_Transformer准备带stacking特征的训练数据
        X_train_stacking = np.concatenate([X_train, self.stacking_train], axis=1)
        X_test_stacking = np.concatenate([X_test, self.stacking_test], axis=1)

        for name, model in tqdm(self.models.items(), desc="训练模型"):
            try:
                # 根据模型选择合适的训练数据
                if name == 'Stacking_Transformer':
                    train_data = X_train_stacking
                    test_data = X_test_stacking
                else:
                    train_data = X_train
                    test_data = X_test

                pipeline = Pipeline([('model', model)])
                grid_search = GridSearchCV(
                    pipeline,
                    self.param_grids.get(name, {}),
                    cv=3,
                    scoring='r2',
                    n_jobs=1
                )
                grid_search.fit(train_data, self.y_train.values)
                y_pred = grid_search.predict(test_data)

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

                print(f"{name} 训练完成，平均R2分数: {avg_r2:.4f}")

            except Exception as e:
                print(f"{name} 训练出错: {str(e)}")
                continue

        # 保存最佳模型
        torch.save(self.best_model, 'models/best_transformer_model.pth')
        print(f"最佳模型是 {self.best_model_name}，平均R2分数: {self.best_model_score:.4f}")

    def evaluate_models(self):
        summary = []
        for model_name, results in self.evaluation_results.items():
            # 验证结果数据是否完整
            if 'average_metrics' not in results:
                print(f"警告: 模型 {model_name} 没有有效的评估指标")
                continue

            summary.append({
                '模型名称': model_name,
                '平均R2分数': results['average_metrics']['r2'],
                '平均MSE': results['average_metrics']['mse'],
                '平均MAE': results['average_metrics']['mae']
            })

        # 检查是否有有效的评估结果
        if not summary:
            print("错误: 没有生成任何模型评估结果，请检查训练过程")
            return None

        try:
            summary_df = pd.DataFrame(summary).sort_values(by='平均R2分数', ascending=False)
            summary_df.to_csv('results/model_evaluation_summary.csv', index=False)

            plt.figure(figsize=(12, 6))
            sns.barplot(x='模型名称', y='平均R2分数', data=summary_df)
            plt.title('不同模型的平均R2分数对比')
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
            plt.title('各目标变量的最佳R2分数')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig('results/target_best_r2.png')
            plt.close()

            return summary_df
        except KeyError as e:
            print(f"排序时发生错误: {e}")
            print("当前的评估结果数据:")
            print(pd.DataFrame(summary).columns)  # 打印实际的列名
            return pd.DataFrame(summary)

    def predict(self, new_data_path):
        new_data = pd.read_csv(new_data_path)
        X_new = new_data[self.feature_columns]
        X_new_processed = self.preprocessor.transform(X_new)

        # 如果是stacking模型，需要生成stacking特征
        if self.best_model_name == 'Stacking_Transformer':
            # 为新数据生成stacking特征
            kf = KFold(n_splits=5, shuffle=False)  # 预测时不打乱数据
            stacking_features = []

            for clf in self.stacking_models:
                _, test_data, _ = clf(
                    self.clean_data(self.preprocessor.transform(self.X)),
                    self.y.values,
                    self.clean_data(X_new_processed),
                    kf
                )
                stacking_features.append(test_data)

            stacking_new = np.concatenate(stacking_features, axis=1)
            X_new_processed = np.concatenate([X_new_processed, stacking_new], axis=1)

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

    # 获取特征列和目标列
    feature_columns = [col for col in new_data.columns if col != 'segment_id']
    target_columns = [col for col in new_data.columns if
                      col.startswith('cutter_') or col in ['energy', 'advance_speed_mm_per_min_mean']]
    target_columns = [col for col in target_columns if col in new_data.columns]

    X_new = new_data[feature_columns]
    X_new_processed = preprocessor.transform(X_new)

    # 检查是否是stacking模型
    if hasattr(best_model, 'named_steps') and 'Stacking' in best_model.named_steps['model'].__class__.__name__:
        # 加载训练数据以生成stacking特征
        data = pd.read_excel("./data/merged_tunnel_data_new.xlsx").fillna(0)
        X_all = data[[col for col in feature_columns if col in data.columns]]
        y_all = data[target_columns] if target_columns else None

        # 生成stacking特征
        kf = KFold(n_splits=5, shuffle=False)
        stacking_models = [lgb_reg, xgb_reg, rf_reg, et_reg]
        stacking_features = []

        for clf in stacking_models:
            _, test_data, _ = clf(
                preprocessor.transform(X_all),
                y_all.values,
                X_new_processed,
                kf
            )
            stacking_features.append(test_data)

        stacking_new = np.concatenate(stacking_features, axis=1)
        X_new_processed = np.concatenate([X_new_processed, stacking_new], axis=1)

    predictions = best_model.predict(X_new_processed)
    predictions_df = pd.DataFrame(predictions, columns=target_columns)
    result_df = pd.concat([new_data['segment_id'], predictions_df], axis=1)
    return result_df


if __name__ == "__main__":
    data_path = "./data/merged_tunnel_data_new.xlsx"
    tbm_model = TBMMultiOutputTransformerModel(data_path)
    tbm_model.init_preprocessor()
    tbm_model.train_models()
    tbm_model.evaluate_models()
