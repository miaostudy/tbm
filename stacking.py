import pandas as pd
import numpy as np
import xgboost
import lightgbm
from sklearn.model_selection import KFold
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 定义stacking回归函数, 根据传入的回归器的名字来设置对应的参数，然后训练
def stacking_reg(clf, train_x, train_y, test_x, clf_name, kf, label_split=None):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((kf.get_n_splits(), test_x.shape[0], 1))
    cv_scores = []

    for i, (train_index, test_index) in enumerate(kf.split(train_x, label_split)):
        tr_x, tr_y = train_x[train_index], train_y[train_index]
        te_x, te_y = train_x[test_index], train_y[test_index]

        if clf_name in ["rf", "ada", "gb", "et", "lr"]:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["xgb"]:
            train_matrix = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_matrix = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, missing=-1)

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

            model = clf.train(
                            params, train_matrix,
                              num_boost_round=10000,
                              evals=[(train_matrix, 'train'), (test_matrix, 'eval')],
                              verbose_eval=False)

            pre = model.predict(test_matrix).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = model.predict(z).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ["lgb"]:
            train_matrix = clf.Dataset(tr_x, label=tr_y)
            test_matrix = clf.Dataset(te_x, label=te_y, reference=train_matrix)

            params = {
                'boosting_type': 'gbdt',
                'objective': 'regression_l2',
                'metric': 'mse',
                'min_child_weight': 1.5,
                'num_leaves': 2 ** 5,
                'lambda_l2': 10,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7,
                'learning_rate': 0.03,
                'tree_method': 'exact',
                'seed': 2017,
                'nthread': 12,
                'silent': True,
            }

            model = clf.train(params, train_matrix,
                              num_boost_round=10000,
                              valid_sets=test_matrix,
                              verbose_eval=False)

            pre = model.predict(te_x, num_iteration=model.best_iteration).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = model.predict(test_x, num_iteration=model.best_iteration).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError("请添加新的分类器")

        print(f"{clf_name} 第{i + 1}折得分: {cv_scores[-1]}")

    test[:] = test_pre.mean(axis=0)
    print(f"{clf_name} 交叉验证得分: {cv_scores}")
    print(f"{clf_name} 平均得分: {np.mean(cv_scores)}")
    return train.reshape(-1, 1), test.reshape(-1, 1)


# 定义各种回归模型的stacking函数
def rf_reg(x_train, y_train, x_valid, kf, label_split=None):
    randomforest = RandomForestRegressor(n_estimators=600, max_depth=20,
                                         n_jobs=-1, random_state=2017,
                                         verbose=0)
    return stacking_reg(randomforest, x_train, y_train, x_valid, "rf", kf, label_split), "rf_reg"


def ada_reg(x_train, y_train, x_valid, kf, label_split=None):
    adaboost = AdaBoostRegressor(n_estimators=30, random_state=2017, learning_rate=0.01)
    return stacking_reg(adaboost, x_train, y_train, x_valid, "ada", kf, label_split), "ada_reg"


def gb_reg(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingRegressor(learning_rate=0.04, n_estimators=100,
                                     subsample=0.8, random_state=2017,
                                     max_depth=5, verbose=0)
    return stacking_reg(gbdt, x_train, y_train, x_valid, "gb", kf, label_split), "gb_reg"


def et_reg(x_train, y_train, x_valid, kf, label_split=None):
    extratree = ExtraTreesRegressor(n_estimators=600, max_depth=35,
                                     n_jobs=-1,
                                    random_state=2017, verbose=0)
    return stacking_reg(extratree, x_train, y_train, x_valid, "et", kf, label_split), "et_reg"


def lr_reg(x_train, y_train, x_valid, kf, label_split=None):
    lr = LinearRegression(n_jobs=-1)
    return stacking_reg(lr, x_train, y_train, x_valid, "lr", kf, label_split), "lr_reg"


def xgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    return stacking_reg(xgboost, x_train, y_train, x_valid, "xgb", kf, label_split), "xgb_reg"


def lgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    return stacking_reg(lightgbm, x_train, y_train, x_valid, "lgb", kf, label_split), "lgb_reg"


# 数据预处理函数
def preprocess_data(data):
    # 处理缺失值和无穷值
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(data.mean())
    return data


def main(input_file):
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel(input_file)

    # 定义目标变量
    target_variables = ['energy', 'advance_speed_mm_per_min_mean']
    # 添加所有wear_i_total变量
    wear_targets = [col for col in df.columns if col.startswith('wear_') and col.endswith('_total')]
    target_variables.extend(wear_targets)

    print(f"检测到的目标变量: {target_variables}")

    # 配置stacking
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=0)

    # 选择要使用的回归模型
    clf_list = [rf_reg, ada_reg, gb_reg, et_reg, lr_reg, xgb_reg, lgb_reg]

    # 为每个目标变量生成文件
    for target in target_variables:
        print(f"\n处理目标变量: {target}")

        # 确保目标变量存在
        if target not in df.columns:
            print(f"警告: 目标变量 {target} 不在数据中，已跳过")
            continue

        # 确定特征列（排除所有目标变量）
        feature_columns = [col for col in df.columns if col not in target_variables]

        # 准备数据
        X = df[feature_columns].copy()
        y = df[target].copy()

        # 预处理数据
        X = preprocess_data(X)

        # 转换为numpy数组
        x_train = X.values
        y_train = y.values
        x_valid = x_train  # 使用整个数据集作为验证集，因为我们只是生成特征

        # 生成stacking特征
        print("正在生成stacking特征...")
        train_data_list = []
        column_names = []

        for clf in clf_list:
            try:
                (train_data, _), clf_name = clf(x_train, y_train, x_valid, kf)
                train_data_list.append(train_data)
                column_names.append(clf_name)
            except Exception as e:
                print(f"模型 {clf.__name__} 出错: {str(e)}")
                continue

        # 合并stacking特征
        if train_data_list:
            stacking_features = np.concatenate(train_data_list, axis=1)
            stacking_df = pd.DataFrame(stacking_features, columns=column_names)

            # 合并原始特征和stacking特征以及目标变量
            result_df = pd.concat([X.reset_index(drop=True),
                                   stacking_df.reset_index(drop=True),
                                   y.reset_index(drop=True).rename(target)], axis=1)

            # 保存为Excel文件
            output_file = f"{target}.xlsx"
            result_df.to_excel(output_file, index=False)
            print(f"已保存文件: {output_file}")
        else:
            print(f"警告: 没有生成任何stacking特征，跳过目标变量 {target}")


if __name__ == "__main__":
    input_file = "data/merged_tunnel_data_new.xlsx"
    main(input_file)
