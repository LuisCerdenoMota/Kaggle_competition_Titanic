import pandas as pd

import functions.eda as eda
import functions.preprocessing as pp
from functions.ml_training import MLTraining, ModelEvaluation


# https://www.kaggle.com/competitions/titanic
TEST_DATA_FILE = "data/test.csv"
TRAIN_DATA_FILE = "data/train.csv"
SUBMISSION_FILE_NAME = "data/gender_submission.csv"
PS_SUBMISSION_FILE_NAME = "data/perfect_score_submission.csv"
N_SPLITS = 5


def generate_submission_file(probabilities):
    class_survived = [col for col in probabilities.columns if col.endswith('Prob_1')]
    probabilities['1'] = probabilities[class_survived].sum(axis=1) / N_SPLITS
    probabilities['0'] = probabilities.drop(columns=class_survived).sum(axis=1) / N_SPLITS
    probabilities['pred'] = 0
    pos = probabilities[probabilities['1'] >= 0.5].index
    probabilities.loc[pos, 'pred'] = 1

    y_pred = df_probabilities['pred'].astype(int)

    submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
    submission['PassengerId'] = df_test['PassengerId']
    submission['Survived'] = y_pred.values
    submission.to_csv(SUBMISSION_FILE_NAME, index=False)
    print(f"Submission file saved as {SUBMISSION_FILE_NAME}")

    return submission


def obtain_true_accuracy(submission):
    df_perfect_score = pd.read_csv(PS_SUBMISSION_FILE_NAME)
    df_perfect_score.rename(columns={'Survived': 'Survived_perfect'}, inplace=True)
    submission.rename(columns={'Survived': 'Survived_test'}, inplace=True)

    df_comparaison = df_perfect_score.merge(submission, how='left', on='PassengerId')
    df_comparaison.loc[df_comparaison['Survived_test'] == df_comparaison['Survived_perfect'], 'comp'] = 1
    df_comparaison.loc[df_comparaison['Survived_test'] != df_comparaison['Survived_perfect'], 'comp'] = 0

    return df_comparaison['comp'].sum() / len(df_comparaison)


if __name__ == "__main__":
    df_train_data = pd.read_csv(TRAIN_DATA_FILE)
    df_train_data['origin'] = 'train'
    df_test_data = pd.read_csv(TEST_DATA_FILE)
    df_test_data['origin'] = 'test'
    df_all_data = pd.concat([df_train_data, df_test_data], sort=True).reset_index(drop=True)

    # Exploratory data analysis (EDA):
    eda.explore_dataset(df_all_data)
    df = eda.explore_variables(df_all_data)
    df = pp.feature_engineering(df)
    df_train, df_test = pp.preprocessing(df)

    # Initialize the class:
    ml_class = MLTraining(df_train, df_test, N_SPLITS)

    # Train and evaluate the model:
    params, oob_score, models, df_fea_importances, list_fprs, list_tprs, df_probabilities = ml_class.train_model()

    model_evaluate_class = ModelEvaluation(params, oob_score, models, df_fea_importances,
                                           list_fprs, list_tprs, df_probabilities)
    model_evaluate_class.feature_importance()
    model_evaluate_class.plot_roc_curve()

    df_submission = generate_submission_file(df_probabilities)

    true_accurate = obtain_true_accuracy(df_submission)
    print(f"The accuracy of the model in kaggle will be: {true_accurate}")

    model_evaluate_class.save_into_mlflow(true_accurate)
