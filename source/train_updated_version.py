import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Define the model structure
from sklearn.utils import shuffle


class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.fc(x)



def transform_label(value):
    if value == 6:
        return 0
    elif value == 5 :
        return 0
    else:
        return 1

def iterabel_dataset_generation(path):
    df = pd.read_csv(path)
    df['new_label'] = df['new_label'].apply(transform_label)

    # Normalize the data
    index = df['new_label'].values

    data = df.iloc[:, :-1]
    num_rows, num_cols = data.shape
    rows_per_chunk = num_rows // 10
    cols_per_chunk = num_cols // 10

    for i in range(9):
        start_row = i * rows_per_chunk
        end_row = start_row + rows_per_chunk
        end_col = (i + 1) * cols_per_chunk

        # 将超出范围的列设置为 0
        data.iloc[start_row:end_row, end_col:] = 0

    # 处理最后一个块（它可能有多于 rows_per_chunk 的行）
    data.iloc[9 * rows_per_chunk:, :] = data.iloc[9 * rows_per_chunk:, :]
    #dataset = Dataset.from_pandas(data)
    label = df.iloc[:, -1].values.astype(int)

    data, label, index = shuffle(data, label, index)
    #iterable_dataset = dataset.to_iterable_dataset()
    return data.values, label, index

# Train the model, one sample at a time

# Create the model, criterion, and optimizer
input_dim = 94  # Adjust this as needed
model = BinaryClassifier(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

correct_predictions = 0
path = '../data/generated_negtive_oversample_data.csv'
inputs, labels, _ = iterabel_dataset_generation(path)
inputs, labels = torch.tensor(inputs), torch.tensor(labels)


for i in range(len(labels)):
    input = inputs[i].float() # Add batch dimension
    label = labels[i].float().unsqueeze(0)  # Add batch dimension

    # Forward pass
    output = model(input)

    # Compute loss
    loss = criterion(output, label)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Predictions for accuracy calculation
    predicted = (output > 0.5).float()
    if predicted == label:
        correct_predictions += 1

# Calculate and print accuracy
accuracy = correct_predictions / len(labels)
print(f"Accuracy: {accuracy * 100:.2f}%")
