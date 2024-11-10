import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

def convert_to_int(word):
    word_dict = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'zero': 0, 0: 0
    }
    return word_dict[word]

if __name__ == "__main__":
    dataset = pd.read_csv("C:\\Users\\hp\\Desktop\\College\\modeldeployment\\tut2\\hiring.csv")
    dataset['Experience'] = dataset['Experience'].fillna(0)
    dataset['test_score'] = dataset['test_score'].fillna(dataset['test_score'].mean())

    # Selecting the first three columns for X and the last column for y
    X = dataset.iloc[:, :3]
    X['Experience'] = X['Experience'].apply(lambda x: convert_to_int(x))
    y = dataset.iloc[:, -1]

    # Train the model
    regressor = LinearRegression()
    regressor.fit(X, y)

    # Save the model
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(regressor, model_file)
        
    model = pickle.load(open('model.pkl', 'rb'))
    # Define feature names to match the training data
    prediction_input = pd.DataFrame([[1, 7, 7]], columns=['Experience', 'test_score', 'interview_score'])

    # Now make the prediction
    print(model.predict(prediction_input))

