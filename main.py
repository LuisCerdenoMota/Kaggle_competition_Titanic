import pandas as pd

import functions.eda as eda
import functions.ml_training as ml
from functions.mlflow_utils import create_mlflow_experiment

import mlflow.sklearn


# https://www.kaggle.com/competitions/titanic
TEST_DATA_FILE = "data/test.csv"
TRAIN_DATA_FILE = "data/train.csv"
SUBMISSION_FILE_NAME = "gender_submission.csv"


def generate_submission_file(output_filename, predictions):
    # Create df with the results:
    submission_df = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })

    # Save the df into a csv file:
    submission_df.to_csv(output_filename, index=False)
    print(f"Submission file saved as {output_filename}")


if __name__ == "__main__":
    train_data = pd.read_csv(TRAIN_DATA_FILE)
    test_data = pd.read_csv(TEST_DATA_FILE)

    # Exploratory data analysis (EDA):
    eda.explore_data(train_data)

    ml_class = ml.MLTraining(train_data, test_data)
    # Preprocessing data:
    preprocessor, x_preprocessed = ml_class.preprocess_data(train=1)

    # Train and evaluate the model:
    best_model, best_accuracy, preprocessor = ml_class.train_and_evaluate(preprocessor, x_preprocessed)
    print(f"Best Model Accuracy: {best_accuracy}")

    # Create an experiment if it's not exists
    experiment_id = create_mlflow_experiment(experiment_name="ML Titanic",
                                             artifact_location="ML_Titanic_artifacts",
                                             tags={"env": "dev", "version": "1.0.0"})
    # Save data and model in ML flow:
    with mlflow.start_run(run_name='testing', experiment_id=experiment_id) as run:
        mlflow.log_metric("accuracy", best_accuracy)
        mlflow.sklearn.log_model(best_model, "model")

        run_id = run.info.run_id

    x_test_preprocessed = ml_class.preprocess_test_data(preprocessor)
    predictions = ml.use_model(x_test_preprocessed, run_id)

    # Generate submission file to upload it in kaggle:
    generate_submission_file(SUBMISSION_FILE_NAME, predictions)
