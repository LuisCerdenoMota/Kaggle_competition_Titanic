import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import mlflow.sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import mlflow
from functions.mlflow_utils import create_mlflow_experiment


def create_parameter_grid():
    # LogisticRegression:
    c = [0.1, 0.5, 1, 5, 10]
    penalty = ['l1', 'l2']
    solver = ['liblinear', 'saga']
    max_iter = [int(x) for x in np.linspace(start=50, stop=500, num=50)]
    param_logisticregression_grid = {
        'C': c,
        'penalty': penalty,
        'solver': solver,
        'max_iter': max_iter
    }

    # RandomForestClassifier:
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=20, stop=2000, num=10)]
    max_features = ["sqrt", "log2"]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 110, num=10)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [int(x) for x in np.linspace(start=2, stop=20, num=2)]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [int(x) for x in np.linspace(start=2, stop=20, num=2)]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    param_randomforest_clas_grid = {
        'n_estimators': n_estimators,  # Número de árboles en el bosque
        'max_features': max_features,  # Número de características a considerar en cada división
        'max_depth': max_depth,  # Profundidad máxima del árbol
        'min_samples_split': min_samples_split,  # Número mínimo de muestras requeridas para dividir un nodo interno
        'min_samples_leaf': min_samples_leaf,  # Número mínimo de muestras requeridas en cada hoja
        'bootstrap': bootstrap  # Método de selección de muestras para entrenar cada árbol
    }

    # SVC:
    kernel = ["linear", "poly", "rbf", "sigmoid"]
    gamma = ['scale', 'auto']
    param_svc_grid = {
        'C': c,
        'kernel': kernel,
        'gamma': gamma
    }

    # KNeighborsClassifier:
    n_neighbors = [3, 5, 7]
    weights = ['uniform', 'distance']
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    param_knclas_grid = {
        'n_neighbors': n_neighbors,
        'weights': weights,
        'algorithm': algorithm
    }

    learning_rate = np.arange(0.001, 0.51, 0.05)
    param_gbc_grid = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate
    }

    param_lgbm_grid = {
        'n_estimators': n_estimators,  # Number of boosting rounds
        'learning_rate': [0.05, 0.1, 0.5],  # Boosting learning rate
        'num_leaves': [20, 30, 40],  # Maximum tree leaves for base learners
        'max_depth': [-1, 5, 10],  # Maximum tree depth for base learners, -1 means no limit
        'min_child_samples': [10, 20, 30],  # Minimum number of data needed in a child (leaf)
        'subsample': [0.8, 0.9, 1.0],  # Subsample ratio of the training instances
        'colsample_bytree': [0.8, 0.9, 1.0],  # Subsample ratio of columns when constructing each tree
        'reg_alpha': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9],  # L1 regularization term on weights
        'reg_lambda': [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]  # L2 regularization term on weights
    }

    grid = {
        'LogisticRegression': param_logisticregression_grid,
        'RandomForestClassifier': param_randomforest_clas_grid,
        'SVC': param_svc_grid,
        'KNeighborsClassifier': param_knclas_grid,
        'GradientBoostingClassifier': param_gbc_grid,
        'LGBMClassifier': param_lgbm_grid
    }

    return grid


def use_model(preprocessor, run_id):
    model_uri = f'runs:/{run_id}/model'
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Realizar predicciones
    predictions = loaded_model.predict(preprocessor)

    return predictions


class MLTraining:
    def __init__(self, df_train_data, df_test_data, n_splits):
        self.df_train = df_train_data
        self.df_test = df_test_data
        self.n_splits = n_splits
        self.params = {
            'criterion': 'gini',
            'n_estimators': 1750,
            'max_depth': 7,
            'min_samples_split': 6,
            'min_samples_leaf': 6,
            'max_features': 'sqrt',
            'oob_score': True,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': 1
        }

        """ drop 
                origin: Because we use it to split the train and test datasets.
                Survived: It is the target.
                PassengerId: Not usefull to train
                SibSp and Parch: We use it to create another feature like family_size, ...
                Name: We use it to create another features like family size or survival rates, ...
                Ticket: We use it to create another features like ticket_occurrences, ...
                family: string feature used to create another about survival rates, ...
                Sex, Pclass, Embarked, deck_cabin, title, family_size_grouped: we have this variable split by columns.
                family_size: usefull to create survival rates.
                ticket_survival_rate and family_survival_rate: We use ir to create survival_rate. 
                ticket_survival_rate_NA and family_survival_rate_NA: We use ir to create survival_rate_NA.
            """
        drop_cols = ['origin', 'Survived', 'PassengerId', 'SibSp', 'Parch', 'Name', 'Ticket', 'family',
                     'Sex', 'Pclass', 'Embarked', 'deck_cabin', 'title', 'family_size_grouped', 'family_size',
                     'ticket_survival_rate', 'family_survival_rate', 'ticket_survival_rate_NA',
                     'family_survival_rate_NA']
        self.drop_cols = drop_cols

    def train_model(self):
        df_all_data = pd.concat([self.df_train.drop(columns=self.drop_cols),
                                 self.df_test.drop(columns=self.drop_cols)])
        # We can use:
        #   x_train = df_train.drop(columns=drop_cols).values
        # but testing, I see the model has better performance using StandardScaler
        x_train = StandardScaler().fit_transform(self.df_train.drop(columns=self.drop_cols))
        y_train = self.df_train['Survived'].values
        x_test = StandardScaler().fit_transform(self.df_test.drop(columns=self.drop_cols))
        # print('X_train shape: {}'.format(x_train.shape))
        # print('y_train shape: {}'.format(y_train.shape))
        # print('X_test shape: {}'.format(x_test.shape))

        oob = 0
        models = []
        list_fprs = []
        list_tprs = []
        dict_probatilities = {}
        # Creating a new dataframe where the columns are the folds, the index are the features
        # and its rows has the importances of the features:
        df_fea_importances = pd.DataFrame(np.zeros((x_train.shape[1], self.n_splits)),
                                          columns=['Fold_{}'.format(i) for i in range(1, self.n_splits + 1)],
                                          index=df_all_data.columns)

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        for fold, (train_index, test_index) in enumerate(skf.split(x_train, y_train)):
            print(f"\nFold {fold + 1}")

            x_train_fold, x_test_fold = x_train[train_index], x_train[test_index]
            y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

            model = RandomForestClassifier(**self.params)

            predictor = model.fit(x_train_fold, y_train_fold)
            models.append(predictor)

            train_fpr, train_tpr, train_thresholds = roc_curve(y_train[train_index],
                                                               predictor.predict_proba(x_train[train_index])[:,
                                                               1])
            trn_auc_score = auc(train_fpr, train_tpr)

            test_fpr, test_tpr, test_thresholds = roc_curve(y_train[test_index],
                                                            predictor.predict_proba(x_train[test_index])[:, 1])
            # test_auc_score = auc(test_fpr, test_tpr)
            # scores.append((trn_auc_score, test_auc_score))
            list_fprs.append(test_fpr)
            list_tprs.append(test_tpr)

            dict_probatilities[f'fold_{fold}_prob_0'] = predictor.predict_proba(x_test)[:, 0]
            dict_probatilities[f'fold_{fold}_prob_1'] = predictor.predict_proba(x_test)[:, 1]
            df_fea_importances.iloc[:, fold - 1] = predictor.feature_importances_

            oob += predictor.oob_score_ / self.n_splits
            print(f'Fold {fold + 1} OOB Score: {predictor.oob_score_}')

        print(f'Average OOB Score: {oob}')
        df_probabilities = pd.DataFrame(dict_probatilities)

        return self.params, oob, models, df_fea_importances, list_fprs, list_tprs, df_probabilities


class ModelEvaluation:
    def __init__(self, params, oob_score, models, df_fea_importances, list_fprs, list_tprs, df_probabilities):
        self.params = params
        self.oob_score = oob_score
        self.models = models
        self.df_fea_importances = df_fea_importances
        self.list_fprs = list_fprs
        self.list_tprs = list_tprs
        self.df_probabilities = df_probabilities

    def feature_importance(self):
        self.df_fea_importances['mean'] = self.df_fea_importances.mean(axis=1)
        self.df_fea_importances.sort_values(by='mean', inplace=True, ascending=False)

        plt.figure(figsize=(15, 20))
        sns.barplot(x='mean', y=self.df_fea_importances.index, data=self.df_fea_importances)

        plt.xlabel('')
        plt.tick_params(axis='x', labelsize=15)
        plt.tick_params(axis='y', labelsize=15)
        plt.title('Random Forest feature importance between folds', size=15)

        plt.savefig('randomforest_feature_importance.png')

    def plot_roc_curve(self):
        tprs_interp = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        f, ax = plt.subplots(figsize=(15, 15))

        # Plotting ROC for each fold and computing AUC scores
        for i, (fpr, tpr) in enumerate(zip(self.list_fprs, self.list_tprs), 1):
            tprs_interp.append(np.interp(mean_fpr, fpr, tpr))
            tprs_interp[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            ax.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC Fold {} (AUC = {:.3f})'.format(i, roc_auc))

        # Plotting ROC for random guessing
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8, label='Random Guessing')

        mean_tpr = np.mean(tprs_interp, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Plotting the mean ROC
        ax.plot(mean_fpr, mean_tpr, color='b',
                label='Mean ROC (AUC = {:.3f} $\pm$ {:.3f})'.format(mean_auc, std_auc),
                lw=2,
                alpha=0.8)

        # Plotting the standard deviation around the mean ROC Curve
        std_tpr = np.std(tprs_interp, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper,
                        color='grey', alpha=.2,
                        label='$\pm$ 1 std. dev.')

        ax.set_xlabel('FPR', size=15, labelpad=20)
        ax.set_ylabel('TPR', size=15, labelpad=20)
        ax.tick_params(axis='x', labelsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])

        ax.set_title('ROC curves of folds', size=20, y=1.02)
        ax.legend(loc='lower right', prop={'size': 13})

        plt.savefig('ROC_curves_between_folds.png')

    def save_into_mlflow(self, true_accurate):
        all_png_images = [x for x in os.listdir(os.getcwd()) if x.endswith('.png')]
        # Create an experiment if it's not exists
        experiment_id = create_mlflow_experiment(experiment_name="ML Titanic",
                                                 artifact_location="ML_Titanic_artifacts",
                                                 tags={"env": "dev", "version": "1.0.0"})
        # Save data and model in ML flow:
        with mlflow.start_run(run_name='RandomForest with feature engineering', experiment_id=experiment_id) as run:
            metrics = {
                "Average OOB score": self.oob_score,
                "True Accurate": true_accurate
            }
            mlflow.log_metrics(metrics)
            mlflow.log_params(self.params)

            for i, model in enumerate(self.models):
                model_name = f"RandomForestModel_fold_{i + 1}"
                mlflow.sklearn.log_model(model, model_name)

            for file in all_png_images:
                mlflow.log_artifact(file)
                os.remove(file)
