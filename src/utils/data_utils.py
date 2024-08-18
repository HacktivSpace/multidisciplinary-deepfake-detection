import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessing the input data by filling missing values, normalizing features, etc.
    :param data: DataFrame containing the input data
    :return: Preprocessed DataFrame
    """
    data = data.fillna(0)
    
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numeric_features] = (data[numeric_features] - data[numeric_features].mean()) / data[numeric_features].std()
    categorical_features = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_features)
    
    return data

def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Splitting the data into training and test sets.
    :param data: DataFrame containing the input data
    :param test_size: Proportion of the data to include in the test set
    :param random_state: Seed used by the random number generator
    :return: Tuple containing training and test sets
    """
    labels = data['label']
    features = data.drop('label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

def balance_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Balancing the dataset by oversampling the minority class.
    :param data: DataFrame containing the input data
    :return: Balanced DataFrame
    """
    from sklearn.utils import resample

    
    majority_class = data[data['label'] == 0]
    minority_class = data[data['label'] == 1]
    
    # To upsample minority class
    minority_upsampled = resample(minority_class, 
                                  replace=True,  
                                  n_samples=len(majority_class),    
                                  random_state=42) 
    
    upsampled_data = pd.concat([majority_class, minority_upsampled])
    
    return upsampled_data

if __name__ == "__main__":
    sample_data = {
        'feature1': [1, 2, 3, 4, 5, 6],
        'feature2': ['A', 'B', 'A', 'A', 'B', 'B'],
        'label': [0, 1, 0, 0, 1, 1]
    }
    df = pd.DataFrame(sample_data)
    
    preprocessed_df = preprocess_data(df)
    print("Preprocessed Data:\n", preprocessed_df)
    
    X_train, X_test, y_train, y_test = split_data(preprocessed_df)
    print("Training Features:\n", X_train)
    print("Test Features:\n", X_test)
    print("Training Labels:\n", y_train)
    print("Test Labels:\n", y_test)
    
    balanced_df = balance_data(df)
    print("Balanced Data:\n", balanced_df)
