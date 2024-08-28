# Import necessary libraries
from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import tensorflow as tf
from flask_cors import CORS
import pickle
from pymongo import MongoClient  # Import pymongo for MongoDB

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# MongoDB connection setup
mongo_uri = "mongodb+srv://thushanvithana:sIdKc9WD0sV6GMIb@cluster0.z9omocd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client['retinaData']  # Replace with your database name

# Define collections
retina_collection = db['retinaData']  # Collection for retinopathy data
diabetic_collection = db['diabeticData']  # Collection for diabetic data

# Path to saved model
saved_model_path = 'trained_model.sav'
# Load the trained model from file
loaded_model = pickle.load(open(saved_model_path, "rb"))

# Function to predict diabetes outcome
def predict_outcome(new_data):
    new_data = np.array([new_data])
    prediction = loaded_model.predict(new_data)[0][0] 
    return prediction


# Endpoint to fetch all diabetes data
@app.route('/diabetes-data', methods=['GET'])
def get_diabetes_data():
    diabetes_data = list(diabetic_collection.find({}, {'_id': 0}))  # Fetch all diabetes data
    return jsonify(diabetes_data)

# Endpoint to fetch all retinopathy data
@app.route('/retinopathy-data', methods=['GET'])
def get_retinopathy_data():
    retinopathy_data = list(retina_collection.find({}, {'_id': 0}))  # Fetch all retinopathy data
    return jsonify(retinopathy_data)



# Function to train KNN model for retinopathy prediction
def train_knn_model(data):
    # Preprocessing: Calculate 'Years Since Diagnosis'
    data['Years Since Diagnosis'] = 2024 - data['Diagnosis Year']
    # Separate features and target variable
    X = data.drop(['Retinopathy Status', 'Retinopathy Probability', 'Diagnosis Year'], axis=1) 
    y = data['Retinopathy Probability'] 
    y = y.round().astype(int)
    # Define categorical and numerical features
    categorical_features = ['Gender', 'Diabetes Type']
    numerical_features = list(set(X.columns) - set(categorical_features))
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)])
    # Define KNN classifier
    knn_classifier = KNeighborsClassifier()
    # Create a pipeline with preprocessing and classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', knn_classifier)])
    # Define hyperparameters grid for GridSearchCV
    param_grid = {'classifier__n_neighbors': [3, 5, 7, 9, 11],
                  'classifier__weights': ['uniform', 'distance'],
                  'classifier__metric': ['euclidean', 'manhattan'],
                  'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'classifier__leaf_size': [20, 30, 40, 50],
                  'classifier__p': [1, 2]}
    # Perform grid search for hyperparameter tuning
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    # Get best parameters and best pipeline
    best_params = grid_search.best_params_
    best_pipeline = grid_search.best_estimator_
    return best_pipeline

# Function to predict retinopathy using KNN model
def predict_with_knn(model, input_data):
    predicted_labels = model.predict(input_data)
    retinopathy_probabilities = [0, 1]
    predicted_probabilities = [retinopathy_probabilities[label] for label in predicted_labels]
    return predicted_probabilities

# Endpoint to predict diabetes outcome
@app.route('/predict-diabetes', methods=['POST'])
def predict():
    data = request.json.get('data')
    if data is None:
        return jsonify({"error": "Data not provided"}), 400
    data_tpl = tuple(data)
    input_data_as_numpy_array = np.asarray(data_tpl)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    # Convert NumPy data types to Python native types
    python_prediction = int(prediction[0])  # Convert numpy.int64 to int
    
    # Insert the input data and prediction into diabeticData collection
    diabetic_collection.insert_one({"data": data, "prediction": python_prediction})
    
    if python_prediction == 0:
        return jsonify({"summary": 'Diabetes negative'})
    else:
        return jsonify({"summary": 'Diabetes positive'})

# Load dataset and train KNN model
data = pd.read_csv('dataset/diabetes_retinopathy_dataset.csv')
knn_model = train_knn_model(data)

# Endpoint to predict retinopathy
@app.route('/predict-retinopathy', methods=['POST'])
def predict_knn():
    data = request.json
    data = data.get("data")
    new_data = pd.DataFrame(data)
    new_data['Years Since Diagnosis'] = 2024 - new_data['Diagnosis Year']
    prediction = predict_with_knn(knn_model, new_data)
    
    # Insert the input data and prediction into retinaData collection
    retina_collection.insert_one({"data": data, "prediction": prediction})
    
    return jsonify({'prediction': "Negative" if prediction[0] == 0 else "Positive"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
