import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to train a KNN model using GridSearchCV
def train_knn_model(data):
    # Calculate 'Years Since Diagnosis' based on 'Diagnosis Year'
    data['Years Since Diagnosis'] = 2024 - data['Diagnosis Year']
    
    # Separate features and target variable
    X = data.drop(['Retinopathy Status', 'Retinopathy Probability', 'Diagnosis Year'], axis=1) 
    y = data['Retinopathy Probability'] 
    # Round and convert probabilities to integers
    y = y.round().astype(int)

    # Define categorical and numerical features
    categorical_features = ['Gender', 'Diabetes Type']
    numerical_features = list(set(X.columns) - set(categorical_features))

    # Preprocess features using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)])

    # KNN classifier
    knn_classifier = KNeighborsClassifier()

    # Create a pipeline for preprocessing and classification
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', knn_classifier)])

    # Define hyperparameter grid for GridSearchCV
    param_grid = {'classifier__n_neighbors': [3, 5, 7, 9, 11],
                  'classifier__weights': ['uniform', 'distance'],
                  'classifier__metric': ['euclidean', 'manhattan'],
                  'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'classifier__leaf_size': [20, 30, 40, 50],
                  'classifier__p': [1, 2]}

    # Grid search with cross-validation to find the best parameters
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    # Get the best parameters and best pipeline
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(X, y)

    return best_pipeline

# Function to predict retinopathy probabilities using the trained model
def predict_with_knn(model, input_data):
    # Predict labels using the model
    predicted_labels = model.predict(input_data)
    
    # Define retinopathy probabilities for each label
    retinopathy_probabilities = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
    # Map predicted labels to corresponding probabilities
    predicted_probabilities = [retinopathy_probabilities[label] for label in predicted_labels]
    
    return predicted_probabilities

# Read dataset and train KNN model
data = pd.read_csv('dataset/diabetes_retinopathy_dataset.csv')
knn_model = train_knn_model(data)

# New data for prediction
new_data = pd.DataFrame({
    'Gender': ['Male'],
    'Diabetes Type': ['Type 1'],
    'Systolic BP': [120],
    'Diastolic BP': [80],
    'HbA1c (mmol/mol)': [90],
    'Estimated Avg Glucose (mg/dL)': [150],
    'Diagnosis Year': [2014]
})
new_data['Years Since Diagnosis'] = 2024 - new_data['Diagnosis Year']

# Predict retinopathy probabilities for new data
predictions = predict_with_knn(knn_model, new_data)
print("Predictions:", predictions)




