import pandas as pd

# Features in lowercase: New feature
# Features with the first capital letter: Come from the original dataset

"""
PassengerId is the unique id of the row and it doesn't have any effect on target
Survived is the target variable we are trying to predict (0 or 1):
    1 = Survived
    0 = Not Survived
Pclass (Passenger Class) is the socio-economic status of the passenger and it is a categorical ordinal feature which has 3 unique values (1, 2 or 3):
    1 = Upper Class
    2 = Middle Class
    3 = Lower Class
Name, Sex and Age are self-explanatory
SibSp is the total number of the passengers' siblings and spouse
Parch is the total number of the passengers' parents and children
Ticket is the ticket number of the passenger
Fare is the passenger fare
Cabin is the cabin number of the passenger
Embarked is port of embarkation and it is a categorical feature which has 3 unique values (C, Q or S):
    C = Cherbourg
    Q = Queenstown
    S = Southampton
"""


def explore_dataset(data):
    print(f'Number of Training Examples = {data.shape[0]}')
    print(f'Number of Test Examples = {data.shape[0]}\n')
    print(f'Training X Shape = {data.shape}')
    print(f'Training y Shape = {data["Survived"].shape[0]}')
    print(f'Test X Shape = {data.shape}')
    print(f'Test y Shape = {data.shape[0]}')
    print(f"Summary of the dataset:")
    print(data.info())
    print("Statistical summary of the dataset:")
    print(data.describe())
    print("Number of null values of columns with missing values:")
    data2 = data.drop(columns=['Survived'])
    df_null_values = pd.DataFrame(data2.isnull().sum())
    df_null_values.reset_index(inplace=True)
    df_null_values.columns = ['Feature', 'null values']
    print(df_null_values[df_null_values['null values'] > 0])


def explore_variables(df):
    # Age: Filling the missing values with the mean Age:
    df['Age'] = df.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.mean())).reset_index(drop=True)

    # Fare:
    mean_fare = df.groupby(['Pclass', 'Parch', 'SibSp'])['Fare'].mean()[1][0][0]
    df['Fare'] = df['Fare'].fillna(mean_fare)

    # Embarked: Filling the 2 null values with S (Southampton):
    df['Embarked'] = df['Embarked'].fillna('S')

    # Cabin: Cabin colum has missing values, but we will work with this variable in the preprocessing functions,
    # because, we are going to create another column with the deck.

    return df
