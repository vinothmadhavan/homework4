
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb
from .models import load_model, save_model
from .datasets.road_dataset import load_data
# Import your confusion matrix calculation for IoU if available
# from .metrics import ConfusionMatrix

def train(exp_dir="logs", model_name="detector", num_epoch=50, lr=1e-3, batch_size=16, seed=2024):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    model = load_model(model_name)
    model = model.to(device)
    model.train()

    train_data = load_data("drive_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("drive_data/val", shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Add confusion matrix for IoU
    # confusion_matrix = ConfusionMatrix(num_classes=3)

    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0.0
        # Reset confusion matrix
        # confusion_matrix.reset()

        for batch in train_data:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            track = batch['track'].to(device)

            logits, pred_depth = model(img)

            seg_loss = nn.CrossEntropyLoss()(logits, track)
            depth_loss = nn.MSELoss()(pred_depth, depth)

            loss = seg_loss + depth_loss
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update confusion matrix
            # _, preds = torch.max(logits, 1)
            # confusion_matrix.update(preds, track)

        # Compute IoU
        # iou = confusion_matrix.compute_iou()

        model.eval()
        with torch.no_grad():
            for batch in val_data:
                img = batch['image'].to(device)
                depth = batch['depth'].to(device)
                track = batch['track'].to(device)

                logits, pred_depth = model(img)

                _, preds = torch.max(logits, 1)
                accuracy = (preds == track).float().mean().item()

                # Perform MAE calculation if required

        print(f"Epoch {epoch + 1}/{num_epoch}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Log metrics to TensorBoard
        logger.add_scalar('Loss/train', epoch_loss, epoch)
        # logger.add_scalar('IoU', iou, epoch)

    save_model(model)
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1024)

    train(**vars(parser.parse_args()))
