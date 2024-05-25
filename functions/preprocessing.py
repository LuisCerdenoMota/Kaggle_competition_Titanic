import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


# Features in lowercase: New feature
# Features with the first capital letter: Come from the original dataset


def extract_surname(data):
    families = []

    for i in range(len(data)):
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name

        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        for c in string.punctuation:
            family = family.replace(c, '').strip()

        families.append(family)

    return families


def feature_engineering(df):
    # Creating Deck column from the first letter of the Cabin column (M stands for Missing)
    df['deck_cabin'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')
    # Passenger in the T deck is changed to A
    idx = df[df['deck_cabin'] == 'T'].index
    df.loc[idx, 'deck_cabin'] = 'A'
    # A, B and C decks are labeled as ABC because all of them have only 1st class passengers
    # D and E decks are labeled as DE because both of them have similar passenger class distribution and survival rate
    # F and G decks are labeled as FG because of the same reason above
    # M deck doesn't grouped with other decks because it's different from others and has the lowest survival rate.
    df['deck_cabin'] = df['deck_cabin'].replace(['A', 'B', 'C'], 'ABC')
    df['deck_cabin'] = df['deck_cabin'].replace(['D', 'E'], 'DE')
    df['deck_cabin'] = df['deck_cabin'].replace(['F', 'G'], 'FG')
    # We no longer need the cabin feature:
    df.drop(columns=['Cabin'], inplace=True)

    # Due to its distribution we cut the fare and age variable in parts.
    # These variables are now categorical
    df['Fare'] = pd.qcut(df['Fare'], 20)
    df['Age'] = pd.qcut(df['Age'], 10)

    # To see the parts and its distribution:
    for feature in ['Fare', 'Age']:
        fig, axs = plt.subplots(figsize=(20, 9))
        sns.countplot(x=feature, hue='Survived', data=df)

        plt.xlabel('Fare', size=15, labelpad=20)
        plt.ylabel('Passenger Count', size=15, labelpad=20)
        plt.tick_params(axis='x', labelsize=8)
        plt.tick_params(axis='y', labelsize=10)
        plt.xticks(rotation=45)

        plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 10})
        plt.title(f'Survival count in {feature} feature', size=15)
        plt.savefig(f'survival_count_feature_{feature}.png')

    # We create the title feature using the name:
    df['title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]

    # We can quickly create another simple feature, flag is_married
    df['married_flag'] = 0
    df.loc[df['title'] == 'Mrs', 'married_flag'] = 1

    # Plot the title distribution:
    fig, axs = plt.subplots(nrows=2, figsize=(20, 20))
    sns.barplot(x=df['title'].value_counts().index, y=df['title'].value_counts().values, ax=axs[0])

    axs[0].tick_params(axis='x', labelsize=10)
    axs[1].tick_params(axis='x', labelsize=15)

    for i in range(2):
        axs[i].tick_params(axis='y', labelsize=15)

    axs[0].set_title('title value count', size=20)

    # Then, we can group the values:
    df['title'] = df['title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'],
                                      'Miss/Mrs/Ms')
    df['title'] = df['title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'],
                                      'Dr/Military/Noble/Clergy')

    # And plot the distribution and save all in an image.
    sns.barplot(x=df['title'].value_counts().index, y=df['title'].value_counts().values, ax=axs[1])
    axs[1].set_title('Grouped title values counts', size=20)
    plt.savefig('title_distribution_before_after_grouped.png')

    """ Ticket: viewing the data:
    Groups with 2,3 and 4 members -> higher survival rate.
    Passengers who travel alone -> lowest survival rate.
    More than 4 group members -> survival rate decreases.
    """
    # We can create, then, the
    df['ticket_occurrences'] = df.groupby('Ticket')['Ticket'].transform('count')

    fig, axs = plt.subplots(figsize=(15, 10))
    sns.countplot(x='ticket_occurrences', hue='Survived', data=df)

    plt.xlabel('Ocurrences', size=15, labelpad=20)
    plt.ylabel('Count', size=15, labelpad=20)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
    plt.title('Survival count vs ticket ocurrences', size=15)
    plt.savefig('survival_count_vs_ticket_ocurrences.png')

    # Create Family_Size feature using SibSp and Parch and grouped it:
    df['family_size'] = df['SibSp'] + df['Parch'] + 1
    family_map = {1: 'Alone', 2: 'Small', 3: 'Small',
                  4: 'Small', 5: 'Medium', 6: 'Medium',
                  7: 'Large', 8: 'Large', 11: 'Large'}
    df['family_size_grouped'] = df['family_size'].map(family_map)
    df['family'] = extract_surname(df['Name'])

    # We need Survived target, then, we cant continue with df, and we need to split it.
    df_train = df[df['origin'] == 'train']
    df_test = df[df['origin'] == 'test']

    # Creating a list of families and tickets that are occuring in both training and test set
    non_unique_families = [x for x in df_train['family'].unique() if x in df_test['family'].unique()]
    non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

    df_family_survival_rate = df_train[['Survived', 'family', 'family_size']].groupby('family').median()
    df_ticket_survival_rate = df_train[['Survived', 'Ticket', 'ticket_occurrences']].groupby('Ticket').median()

    family_rates = {}
    ticket_rates = {}

    for i in range(len(df_family_survival_rate)):
        # Checking a family exists in both training and test set, and has members more than 1
        if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
            family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

    for i in range(len(df_ticket_survival_rate)):
        # Checking a ticket exists in both training and test set, and has members more than 1
        if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
            ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]

    mean_survival_rate = np.mean(df_train['Survived'])

    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []

    for i in range(len(df_train)):
        if df_train['family'][i] in family_rates:
            train_family_survival_rate.append(family_rates[df_train['family'][i]])
            train_family_survival_rate_NA.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_NA.append(0)

    for i in range(len(df_test)):
        if df_test['family'].iloc[i] in family_rates:
            test_family_survival_rate.append(family_rates[df_test['family'].iloc[i]])
            test_family_survival_rate_NA.append(1)
        else:
            test_family_survival_rate.append(mean_survival_rate)
            test_family_survival_rate_NA.append(0)

    df_train['family_survival_rate'] = train_family_survival_rate
    df_train['family_survival_rate_NA'] = train_family_survival_rate_NA
    df_test['family_survival_rate'] = test_family_survival_rate
    df_test['family_survival_rate_NA'] = test_family_survival_rate_NA

    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []

    for i in range(len(df_train)):
        if df_train['Ticket'][i] in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
            train_ticket_survival_rate_NA.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_NA.append(0)

    for i in range(len(df_test)):
        if df_test['Ticket'].iloc[i] in ticket_rates:
            test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
            test_ticket_survival_rate_NA.append(1)
        else:
            test_ticket_survival_rate.append(mean_survival_rate)
            test_ticket_survival_rate_NA.append(0)

    df_train['ticket_survival_rate'] = train_ticket_survival_rate
    df_train['ticket_survival_rate_NA'] = train_ticket_survival_rate_NA
    df_test['ticket_survival_rate'] = test_ticket_survival_rate
    df_test['ticket_survival_rate_NA'] = test_ticket_survival_rate_NA
    for df in [df_train, df_test]:
        df.loc[:, 'survival_rate'] = (df['ticket_survival_rate'] + df['family_survival_rate']) / 2
        df.loc[:, 'survival_rate_NA'] = (df['ticket_survival_rate_NA'] + df['family_survival_rate_NA']) / 2

    # We concat all the information to finish and return:
    df = pd.concat([df_train, df_test])

    return df


def preprocessing(df):
    # We use labelEncoder only for ordinal features, that is, the values of the feature has an intrinsic order:
    ordinal_features = ['Age', 'Fare']
    for feature in ordinal_features:
        df[feature] = LabelEncoder().fit_transform(df[feature])

    df_train = df[df['origin'] == 'train']
    df_test = df[df['origin'] == 'test']
    list_dfs = [df_train, df_test]

    # We use OneHotEncoder if the variable doesn't have an intrinsic order, or we don't know it.
    cat_features = ['Pclass', 'Sex', 'deck_cabin', 'Embarked', 'title', 'family_size_grouped']
    encoded_features = []
    for df in list_dfs:
        for feature in cat_features:
            encoded_feat = OneHotEncoder().fit_transform(df[[feature]]).toarray()
            n = df[feature].nunique()
            cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
            encoded_df = pd.DataFrame(encoded_feat, columns=cols)
            encoded_df.index = df.index
            encoded_features.append(encoded_df)

    df_train = pd.concat([df_train, *encoded_features[:6]], axis=1)
    df_test = pd.concat([df_test, *encoded_features[6:]], axis=1)

    return df_train, df_test
