import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np


# Define the custom Dataset
class PickleDataset(Dataset):
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)  # data is a list of (input, target) tuples
        # Unzip the list of tuples into separate inputs and targets
        self.data = zip(*data)
        self.X = torch.tensor(self.data[0], dtype=torch.float32).squeeze(-1)
        self.y = torch.tensor(self.data[1], dtype=torch.float32).squeeze(-1)
        print("X", self.X.shape)
        print("y", self.y.shape)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]


# Define the MLP Model
class MLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            # nn.Linear(input_dim, output_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
    average_loss = total_loss / len(data_loader.dataset)
    return average_loss

def eval_cm(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            all_outputs.append(outputs.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    # Thresholding to obtain binary predictions and targets
    threshold = 0.01
    binary_preds = (all_preds > threshold).astype(int).flatten()
    binary_targets = (all_targets > threshold).astype(int).flatten()
    # Compute confusion matrix
    cm = confusion_matrix(binary_targets, binary_preds, labels=[0,1])
    print("CM", cm)
    print("CM", (cm[0,0]+cm[1,1]) / cm.sum())


def find_subgroup_indices(binary_list):
    start = None
    indices = []

    for i, value in enumerate(binary_list):
        if value:
            if start is None:
                start = i  # Start of a new group of 1's
        elif start is not None:
            indices.append((start, i - 1))  # End of the current group of 1's
            start = None  # Reset start for the next group

    if start is not None:  # If a group ends at the end of the list
        indices.append((start, len(binary_list) - 1))

    return indices


def predictions_to_peaks(t, y_pred, threshold):
    for start, end in find_subgroup_indices(y_pred > threshold):
        yield ((t[start] - t[end]) / 2.) + t[start]

def eval_sequence(peak_times, peak_predictions):
    return min_pairing_distance(peak_times, peak_predictions)


def min_pairing_distance(xs, ys):
    n = len(xs)
    m = len(ys)

    xs_sorted = sorted(xs)
    ys_sorted = sorted(ys)

    # Ensure xs is the shorter list to optimize space complexity
    if n > m:
        xs_sorted, ys_sorted = ys_sorted, xs_sorted
        n, m = m, n

    INF = float('inf')

    # Initialize DP table
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        dp[0][j] = 0  # Zero elements in xs, cost is zero

    for i in range(1, n + 1):
        dp[i][0] = INF  # Cannot match xs[0..i-1] with zero elements in ys

    # Fill DP table
    for i in range(1, n + 1):
        for j in range(i, m + 1):  # Ensure j >= i to have enough ys elements to match xs
            cost = abs(xs_sorted[i - 1] - ys_sorted[j - 1])
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,  # Match xs[i-1] with ys[j-1]
                dp[i][j - 1]  # Skip ys[j-1], do not match it
            )

    # The answer is the minimum cost to match all elements in xs
    return dp[n][m]


def main():
    # Hyperparameters
    pickle_file = 'data.pkl'  # Path to your pickle file
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 20
    model_save_path = 'models'  # Path to save the models

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    dataset = PickleDataset(pickle_file)
    example = next(iter(dataset))

    # Split into training and testing (e.g., 80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = MLP(input_dim=20, hidden_dim=64, output_dim=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Calculate and report initial (initialized model) test loss
    initial_test_loss = evaluate_loss(model, test_loader, criterion, device)
    print(f"Initial Test Loss (Before Training): {initial_test_loss:.4f}")
    eval_cm(model, test_loader, device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)

        epoch_loss = running_loss / train_size

        # Evaluate on test set
        test_loss = evaluate_loss(model, test_loader, criterion, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}")
        eval_cm(model, test_loader, device)

        # Save the current model's state_dict
        with open(f"{model_save_path}/model{epoch}.torch", 'wb') as f:
            torch.save(model, f)



if __name__ == "__main__":
    main()
