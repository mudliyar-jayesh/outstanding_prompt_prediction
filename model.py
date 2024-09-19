# train_model.py

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pymongo
import torch
import torch.nn as nn
import torch.optim as optim
import joblib

# Establish MongoDB connection
mongo_client = pymongo.MongoClient("mongodb://softgen:QWAmTnsdBUaTL2z@118.139.167.125:27017/")
database = mongo_client["Summary"]
collection = database["OutstandingSummary"]

cursor = collection.find({})
data = list(cursor)
df = pd.DataFrame(data)

# Handle missing values if any (optional)
df = df.dropna()

print(df[['last_action']])
# One-hot encode 'last_action' feature
action_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
action_encoded = action_encoder.fit_transform(df[['last_action']])
action_encoded_df = pd.DataFrame(action_encoded, columns=action_encoder.get_feature_names_out(['last_action']))

# Encode 'last_outcome' target variable
outcome_encoder = LabelEncoder()
df['last_outcome_encoded'] = outcome_encoder.fit_transform(df['last_outcome'])

# Save the encoders after fitting
joblib.dump(action_encoder, 'action_encoder.joblib')
joblib.dump(outcome_encoder, 'outcome_encoder.joblib')

# Combine the encoded features with the dataframe
df = pd.concat([df, action_encoded_df], axis=1)

# Prepare features (X) and target (y)
feature_columns = ['delay_percentage', 'amount_due', 'average_delay_days'] + list(action_encoded_df.columns)
X = df[feature_columns].values.astype(np.float32)
y = df['last_outcome_encoded'].values.astype(np.float32)

# Split the data into training and testing datasets
X_train, X_test, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert y to tensors
y_train = torch.tensor(y_train_np).unsqueeze(1)
y_test = torch.tensor(y_test_np).unsqueeze(1)

# Define a simple neural network model in PyTorch
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

input_size = X_train.shape[1]
model = NeuralNet(input_size)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert X_train and X_test to tensors
X_train_tensor = torch.tensor(X_train)
X_test_tensor = torch.tensor(X_test)

# Train the model
num_epochs = 100
batch_size = 50 
num_samples = X_train_tensor.shape[0]

for epoch in range(num_epochs):
    model.train()
    permutation = torch.randperm(num_samples)
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train_tensor[indices], y_train[indices]
        
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    predicted = (outputs > 0.5).float()
    correct = (predicted.eq(y_test)).sum().item()
    test_acc = correct / y_test.size(0)
    print(f'Test Accuracy: {test_acc:.2f}')

# Save the model
torch.save(model.state_dict(), 'payment_recommendation_model.pth')

# Save the feature columns
with open('feature_columns.txt', 'w') as f:
    for col in feature_columns:
        f.write(f"{col}\n")

