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
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)
        # No activation function here

    def forward(self, x):
        out = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(out))
        out = self.fc3(out)
        return out

# Load encoders
action_encoder = joblib.load('action_encoder.joblib')
outcome_encoder = joblib.load('outcome_encoder.joblib')

# Ensure the encoder is fitted
try:
    check_is_fitted(action_encoder)
    print("action_encoder is fitted.")
except NotFittedError:
    print("action_encoder is not fitted.")
    exit(1)

# Load feature columns
with open('feature_columns.txt', 'r') as f:
    feature_columns = f.read().splitlines()

# Initialize the model
input_size = len(feature_columns)
num_classes = len(outcome_encoder.classes_)
model = NeuralNet(input_size, num_classes)
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
            last_action = data['last_action'].strip().lower()
        except ValueError:
            return jsonify({'error': 'Invalid input data type'}), 400

        # Preprocess 'last_action' using the loaded OneHotEncoder
        input_df = pd.DataFrame({
            'last_action': [last_action]
        })

        # Check for unknown 'last_action' categories
        if last_action not in action_encoder.categories_[0]:
            return jsonify({'error': f"Unknown 'last_action' category: {last_action}"}), 400

        action_encoded = action_encoder.transform(input_df[['last_action']])
        action_encoded_df = pd.DataFrame(
            action_encoded, 
            columns=action_encoder.get_feature_names_out(['last_action'])
        )

        # Combine numerical features with the one-hot encoded 'last_action'
        numerical_features = pd.DataFrame({
            'delay_percentage': [delay_percentage],
            'amount_due': [amount_due],
            'average_delay_days': [average_delay_days]
        })

        input_processed = pd.concat(
            [numerical_features.reset_index(drop=True), action_encoded_df.reset_index(drop=True)], 
            axis=1
        )

        # Ensure the columns are in the same order as during training
        input_processed = input_processed[feature_columns]

        # Convert to a NumPy array and then to a tensor
        input_tensor = torch.tensor(input_processed.values.astype(np.float32))

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = nn.functional.softmax(output, dim=1)
            top_probs, top_idxs = torch.topk(probabilities, k=3, dim=1)
            top_probs = top_probs.cpu().numpy()[0]
            top_idxs = top_idxs.cpu().numpy()[0]
            top_classes = outcome_encoder.inverse_transform(top_idxs)

            # Prepare the response
            predictions = []
            for cls, prob in zip(top_classes, top_probs):
                predictions.append({
                    'class': cls,
                    'probability': float(prob)
                })

        # Return the top 3 predictions as JSON
        response = {
            'predictions': predictions
        }
        return jsonify(response)

    except Exception as e:
        # Log the exception (optional)
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)

