# app.py

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

# Load the trained model
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.sigmoid(self.fc3(out))
        return out

# Load encoders
action_encoder = joblib.load('action_encoder.joblib')
outcome_encoder = joblib.load('outcome_encoder.joblib')  # Only needed if you want to inverse transform predictions

# Ensure the encoder is fitted
try:
    check_is_fitted(action_encoder)
    print("action_encoder is fitted.")
except NotFittedError:
    print("action_encoder is not fitted.")
    # Handle error appropriately or exit
    exit(1)

# Load feature columns
with open('feature_columns.txt', 'r') as f:
    feature_columns = f.read().splitlines()

# Initialize the model
input_size = len(feature_columns)
model = NeuralNet(input_size)
model.load_state_dict(torch.load('payment_recommendation_model.pth'))
model.eval()

# Create Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        if data is None:
            return jsonify({'error': 'Invalid or missing JSON data'}), 400

        # Required fields
        required_fields = ['delay_percentage', 'amount_due', 'average_delay_days', 'last_action']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'error': f"Missing required fields: {', '.join(missing_fields)}"}), 400

        # Validate and preprocess input data
        try:
            delay_percentage = float(data['delay_percentage'])
            amount_due = float(data['amount_due'])
            average_delay_days = float(data['average_delay_days'])
            last_action = data['last_action']
        except ValueError:
            return jsonify({'error': 'Invalid input data type'}), 400

        # Check for unknown 'last_action' categories
        if last_action not in action_encoder.categories_[0]:
            return jsonify({'error': f"Unknown 'last_action' category: {last_action}"}), 400

        # Create a DataFrame from the input data
        input_df = pd.DataFrame({
            'delay_percentage': [delay_percentage],
            'amount_due': [amount_due],
            'average_delay_days': [average_delay_days],
            'last_action': [last_action]
        })

        # Preprocess 'last_action' using the loaded OneHotEncoder
        action_encoded = action_encoder.transform(input_df[['last_action']])
        action_encoded_df = pd.DataFrame(
            action_encoded, 
            columns=action_encoder.get_feature_names_out(['last_action'])
        )

        # Combine numerical features with the one-hot encoded 'last_action'
        input_processed = pd.concat(
            [input_df[['delay_percentage', 'amount_due', 'average_delay_days']].reset_index(drop=True), action_encoded_df], 
            axis=1
        )

        # Ensure the columns are in the same order as during training
        input_processed = input_processed[feature_columns]

        # Convert to a NumPy array and then to a tensor
        input_tensor = torch.tensor(input_processed.values.astype(np.float32))

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probability = output.item()
            prediction = 1 if probability > 0.5 else 0
            predicted_label = outcome_encoder.inverse_transform([prediction])[0]

        # Return the prediction as JSON
        response = {
            'probability': probability,
            'prediction': predicted_label
        }
        return jsonify(response)

    except Exception as e:
        # Log the exception (optional)
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)

