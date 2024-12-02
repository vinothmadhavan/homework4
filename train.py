import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from homework.models import MLPPlanner  # Assuming MLPPlanner is implemented in models.py
from homework.datasets.road_dataset import RoadDataset  # Assuming RoadDataset is implemented
from homework.metrics import compute_longitudinal_error, compute_lateral_error  # Assuming these are implemented
from homework.utils import save_model  # Assuming save_model is implemented

def train(model, dataloader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in dataloader:
            track_left = batch['track_left']
            track_right = batch['track_right']
            waypoints = batch['waypoints']
            waypoints_mask = batch['waypoints_mask']

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(track_left, track_right)
            loss = criterion(outputs, waypoints)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader)}')

        # Optionally, evaluate the model on validation set and compute metrics
        # val_longitudinal_error, val_lateral_error = evaluate(model, val_dataloader)

    # Save the trained model
    save_model(model, 'mlp_planner.pth')

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10

    # Load dataset
    train_dataset = RoadDataset(split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = MLPPlanner()
    criterion = nn.MSELoss()  # Assuming MSELoss is suitable for waypoint prediction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_dataloader, criterion, optimizer, num_epochs)

if __name__ == '__main__':
    main()
