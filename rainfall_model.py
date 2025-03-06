import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

file_path = '/Users/abhinavsingh/Downloads/rainfall/rainfall.csv'
data = pd.read_csv(file_path)


selected_district = "PATNA"
district_data = data[data['DISTRICT'] == selected_district]


monthly_rainfall = district_data.loc[:, 'JAN':'DEC'].values.flatten()


scaler = MinMaxScaler(feature_range=(0, 1))
norm_rainfall = scaler.fit_transform(monthly_rainfall.reshape(-1, 1))


sequence_length = 6  
if len(norm_rainfall) <= sequence_length:
    raise ValueError(f"Not enough data for the selected sequence_length={sequence_length}. "
                     f"Dataset contains only {len(norm_rainfall)} samples.")


X, y = [], []
for i in range(len(norm_rainfall) - sequence_length):
    X.append(norm_rainfall[i:i + sequence_length])
    y.append(norm_rainfall[i + sequence_length])
X, y = np.array(X), np.array(y)

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)


if X.dim() == 2:
    X = X.unsqueeze(-1)


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

# Define the LSTM Model
class RainfallLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(RainfallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = RainfallLSTM()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the Model
epochs = 40
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(-1))
            val_loss += loss.item()

    val_loss /= len(test_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")


model.eval()
predicted_rainfall = []
actual_rainfall = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        predicted_rainfall.extend(outputs.squeeze().numpy())
        actual_rainfall.extend(y_batch.numpy())

# Inverse transform the predictions and true values
predicted_rainfall = scaler.inverse_transform(np.array(predicted_rainfall).reshape(-1, 1))
actual_rainfall = scaler.inverse_transform(np.array(actual_rainfall).reshape(-1, 1))

# Convert to binary for calculating F1 score, precision, and recall (threshold is chosen arbitrarily, adjust as needed)
threshold = 0.5
binary_actual = (actual_rainfall > threshold).astype(int).flatten()
binary_predicted = (predicted_rainfall > threshold).astype(int).flatten()

# Calculate precision, recall, F1 score, and accuracy
precision = precision_score(binary_actual, binary_predicted)
recall = recall_score(binary_actual, binary_predicted)
f1 = f1_score(binary_actual, binary_predicted)
accuracy = accuracy_score(binary_actual, binary_predicted)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Actual vs. Predicted Rainfall
plt.figure(figsize=(12, 6))
plt.plot(actual_rainfall, label='Actual Rainfall')
plt.plot(predicted_rainfall, label='Predicted Rainfall')
plt.title('Actual vs Predicted Rainfall')
plt.xlabel('Time Steps')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()

# Add a summary table of metrics
summary_table = pd.DataFrame({
    "Metric": ["Precision", "Recall", "F1 Score", "Accuracy"],
    "Value": [precision, recall, f1, accuracy]
})

print("\nSummary Table of Evaluation Metrics:")
print(summary_table)

# Save the model for future use
torch.save(model.state_dict(), "rainfall_forecasting_model.pth")
print("Rainfall forecasting model trained and saved successfully!")
