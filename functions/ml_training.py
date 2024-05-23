from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import mlflow


def use_model(preprocessor, run_id):
    model_uri = f'runs:/{run_id}/model'
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # Realizar predicciones
    predictions = loaded_model.predict(preprocessor)

    return predictions


class MLTraining:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.x = train_data.drop(columns=["Survived", "PassengerId"])
        self.y = train_data["Survived"]
        self.x_test_data = test_data.drop(columns=["PassengerId"])

    def preprocess_data(self, train):
        # Split data into numerical and categorical:
        numeric_features = self.x.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.x.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        x_preprocessed = preprocessor.fit_transform(self.x)

        return preprocessor, x_preprocessed

    def preprocess_test_data(self, preprocessor):
        x_test_preprocessed = preprocessor.transform(self.x_test_data)

        return x_test_preprocessed

    def train_and_evaluate(self, preprocessor, x_preprocessed):
        x_train, x_test, y_train, y_test = train_test_split(x_preprocessed, self.y, test_size=0.2, random_state=42)

        models = {
            "Logistic Regression": LogisticRegression(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC(),
            "K-Nearest Neighbors": KNeighborsClassifier()
        }

        best_model = None
        best_accuracy = 0.0
        for name, model in models.items():
            """
            # Pipeline with preprocessing and model:
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            """
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} accuracy: {accuracy}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model

        return best_model, best_accuracy, preprocessor
