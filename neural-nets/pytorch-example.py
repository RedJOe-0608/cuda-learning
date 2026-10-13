import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

print("numpy version: ", np.__version__)
print("Torch version: ", torch.__version__)

torch.manual_seed(1)

TRAIN_SIZE = 10000
epochs = 10
learning_rate = 1e-2
batch_size = 8

# precision vs performance tradeoff
torch.set_float32_matmul_precision("high")

# Here we are saying, interpret every 4 bytes as one float32. Because, when we saved into .bin, that file is nothing more than a sequence of bytes. Without .reshape, X_train_np is (47040000,). 
X_train_np = np.fromfile("neural-nets/data/X_train.bin", dtype=np.float32).reshape(60000,784)
y_train_np = np.fromfile("neural-nets/data/y_train.bin", dtype=np.int32)
X_test_np = np.fromfile("neural-nets/data/X_test.bin", dtype=np.float32).reshape(10000,784)
y_test_np = np.fromfile("neural-nets/data/y_test.bin", dtype=np.int32)

mean = X_train_np.mean()
std = X_train_np.std()

X_train_np = (X_train_np - mean) / std
X_test_np = (X_test_np - mean) / std

#moves them to GPU
train_data = torch.from_numpy(X_train_np[:TRAIN_SIZE].reshape(-1,1,28,28)).to("cuda")
train_labels = torch.from_numpy(y_train_np[:TRAIN_SIZE]).long().to("cuda") # convert to long (int64), because, the pytorch's CrossEntropyLoss that we will use expects labels to be int64
test_data = torch.from_numpy(X_test_np.reshape(-1, 1, 28, 28)).to("cuda")
test_labels = torch.from_numpy(y_test_np).long().to("cuda")


class MLP(nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, num_classes)

    def forward(self, x):
        x = x.reshape(batch_size, 28*28) # (1,28,28) -> (batch_size, 28*28) = (8,784)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

model = MLP(784,256,10).to("cuda")

# We are effectively telling pytorch, do not calculate gradients for these operations. We are just initialising weights
with torch.no_grad():
    fan_in_fc1 = model.fc1.weight.size(1)
    scale_fc1 = (6.0 / fan_in_fc1) ** 0.5
    model.fc1.weight.uniform_(-scale_fc1, scale_fc1) # Fill all the weights with random values uniformly distributed between -scale_fc1 and +scale_fc1. This is related to Xavier/Glorot-style initialization.
    model.fc1.bias.zero_() # set all the 256 biases to zero

    fan_in_fc2 = model.fc2.weight.size(1)
    scale_fc2 = (6.0 / fan_in_fc2) ** 0.5
    model.fc2.weight.uniform_(-scale_fc2, scale_fc2)
    model.fc2.bias.zero_()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

def train_timed(model, criterion, optimizer, epoch, timing_stats, epoch_losses):
    """
    Train the model for one epoch with detailed timing instrumentation
    
    Tracks time spent on:
    - Data loading
    - Forward pass
    - Loss computation
    - Backward pass (gradient computation)
    - Weight updates
    
    @param model: PyTorch model to train
    @param criterion: Loss function
    @param optimizer: Optimizer for weight updates
    @param epoch: Current epoch number (unused, kept for compatibility)
    @param timing_stats: Dictionary to accumulate timing statistics
    @param epoch_losses: List to store average loss per epoch
    """
    model.train()  # Set model to training mode
    epoch_loss = 0.0
    
    # Calculate number of iterations per epoch
    iters_per_epoch = math.ceil(train_data.shape[0] / batch_size)
    
    # Process each batch
    for i in range(iters_per_epoch):
        # Time data loading
        data_start = time.time()
        data = train_data[i * batch_size : (i + 1) * batch_size]
        target = train_labels[i * batch_size : (i + 1) * batch_size]
        data_end = time.time()
        timing_stats['data_loading'] += data_end - data_start
        
        # Zero out gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        forward_start = time.time()
        outputs = model(data)
        forward_end = time.time()
        timing_stats['forward'] += forward_end - forward_start
        
        # Compute loss
        loss_start = time.time()
        loss = criterion(outputs, target)
        epoch_loss += loss.item()
        loss_end = time.time()
        timing_stats['loss_computation'] += loss_end - loss_start
        
        # Backward pass: compute gradients
        backward_start = time.time()
        loss.backward()
        backward_end = time.time()
        timing_stats['backward'] += backward_end - backward_start
        
        # Update weights using computed gradients
        update_start = time.time()
        optimizer.step()
        optimizer.zero_grad()  # Zero gradients for next iteration
        update_end = time.time()
        timing_stats['weight_updates'] += update_end - update_start
    
    # Store average loss for this epoch
    epoch_losses.append(epoch_loss / iters_per_epoch)


def evaluate(model, test_data, test_labels):
    """
    Evaluate model accuracy on test dataset
    
    Computes average batch accuracy across all test batches
    Uses no_grad context to disable gradient computation for efficiency
    
    @param model: PyTorch model to evaluate
    @param test_data: Test input data tensor
    @param test_labels: Test labels tensor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    total_batch_accuracy = torch.tensor(0.0, device=device)
    num_batches = 0

    iters = math.ceil(len(test_data) / batch_size)
    # Disable gradient computation during evaluation
    with torch.no_grad():
        for i in range(iters):
            # Get batch data
            data = test_data[i * batch_size : (i + 1) * batch_size]
            target = test_labels[i * batch_size : (i + 1) * batch_size]
            
            # Forward pass to get predictions
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)  # Get predicted class indices
            
            # Compute accuracy for this batch
            correct_batch = (predicted == target).sum().item()
            total_batch = target.size(0)
            if total_batch != 0:  
                batch_accuracy = correct_batch / total_batch
                total_batch_accuracy += batch_accuracy
                num_batches += 1

    # Compute and print average accuracy
    avg_batch_accuracy = total_batch_accuracy / num_batches
    print(f"Average Batch Accuracy: {avg_batch_accuracy * 100:.2f}%")


if __name__ == "__main__":
    # Initialize timing statistics dictionary
    timing_stats = {
        'data_loading': 0.0,
        'forward': 0.0,
        'loss_computation': 0.0,
        'backward': 0.0,
        'weight_updates': 0.0,
        'total_time': 0.0
    }
    epoch_losses = []
    
    # Start total training timer
    total_start = time.time()
    
    # Training loop over epochs
    for epoch in range(epochs):
        train_timed(model, criterion, optimizer, epoch, timing_stats, epoch_losses)
        print(f"Epoch {epoch} loss: {epoch_losses[epoch]:.4f}")

    # Calculate total training time
    total_end = time.time()
    timing_stats['total_time'] = total_end - total_start
    
    # Print detailed timing breakdown
    print("\n=== PYTORCH CUDA IMPLEMENTATION TIMING BREAKDOWN ===")
    print(f"Total training time: {timing_stats['total_time']:.1f} seconds\n")
    
    print("Detailed Breakdown:")
    print(f"  Data loading:     {timing_stats['data_loading']:6.3f}s ({100.0 * timing_stats['data_loading'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Forward pass:     {timing_stats['forward']:6.3f}s ({100.0 * timing_stats['forward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Loss computation: {timing_stats['loss_computation']:6.3f}s ({100.0 * timing_stats['loss_computation'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Backward pass:    {timing_stats['backward']:6.3f}s ({100.0 * timing_stats['backward'] / timing_stats['total_time']:5.1f}%)")
    print(f"  Weight updates:   {timing_stats['weight_updates']:6.3f}s ({100.0 * timing_stats['weight_updates'] / timing_stats['total_time']:5.1f}%)")

    print("Finished Training")