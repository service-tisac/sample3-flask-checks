from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from typing import List

# Define the Neural Network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load model from checkpoint
def load_model(checkpoint_path):
    model = SimpleNN()
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    return model

# Create Flask app
app = Flask(__name__)

# Load the model checkpoint
model = load_model('simple_nn_checkpoint.pth')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/check_model', methods=['GET'])
def model_check():
    try:
        _ = model(torch.randn(1, 10))  # Perform a forward pass with random input
        return jsonify({"status": "model loaded successfully"})
    except Exception as e:
        return jsonify({"status": "model loading failed", "error": str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'input' not in data or len(data['input']) != 10:
        return jsonify({"error": "Input must be a list of 10 floats."}), 400
    
    input_tensor = torch.tensor(data['input']).float().unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)
    
    return jsonify({"output": output.squeeze().item()})

@app.route('/check_inference', methods=['GET'])
def inference_check():
    try:
        input_data = [0.0] * 10  # Dummy input data
        input_tensor = torch.tensor(input_data).float().unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        return jsonify({"status": "inference successful", "output": output.squeeze().item()})
    except Exception as e:
        return jsonify({"status": "inference failed", "error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)