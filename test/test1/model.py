from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        # super().__init__()

        # self.n_track = n_track
        # self.n_waypoints = n_waypoints

        # # Define the MLP layers
        # self.mlp = nn.Sequential(
        #     nn.Linear(n_track * 2 * 2, hidden_size),  # Input size is n_track * 2 (for left and right) * 2 (x, y)
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, n_waypoints * 2)  # Output size is n_waypoints * 2 (x, y)
        # )

        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Define the MLP layers
        # Input size is 2 * n_track * 2 because we concatenate left and right track boundaries
        input_size = 2 * n_track * 2
        hidden_size = 64  # You can adjust the hidden size
        output_size = n_waypoints * 2  # Each waypoint has 2 coordinates (x, y)

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        # Concatenate the left and right track boundaries
        batch_size = track_left.size(0)
        x = torch.cat((track_left, track_right), dim=2)  # Shape: (b, n_track, 4)
        x = x.view(batch_size, -1)  # Flatten to shape: (b, 2 * n_track * 2)

        # Pass through MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape output to (b, n_waypoints, 2)
        waypoints = x.view(batch_size, self.n_waypoints, 2)
        return waypoints
        # x = torch.cat((track_left, track_right), dim=2)  # Shape: (b, n_track, 4)

        # # Flatten the input
        # x = x.view(x.size(0), -1)  # Shape: (b, n_track * 4)

        # # Pass through the MLP
        # waypoints = self.mlp(x)  # Shape: (b, n_waypoints * 2)

        # # Reshape to (b, n_waypoints, 2)
        # waypoints = waypoints.view(-1, self.n_waypoints, 2)

        # return waypoints
        # raise NotImplementedError



# class TransformerPlanner(nn.Module):
#     def __init__(
#         self,
#         n_track: int = 10,
#         n_waypoints: int = 3,
#         d_model: int = 64,
#     ):
#         # super().__init__()

#         # self.n_track = n_track
#         # self.n_waypoints = n_waypoints

#         # self.query_embed = nn.Embedding(n_waypoints, d_model)

#         super().__init__()

#         self.n_track = n_track
#         self.n_waypoints = n_waypoints
#         self.d_model = d_model

#         # Embedding for the query (waypoints)
#         self.query_embed = nn.Embedding(n_waypoints, d_model)

#         # Linear layer to project input features to d_model
#         self.input_proj = nn.Linear(4, d_model)  # 4 because we concatenate left and right (2+2)

#         # Transformer decoder
#         decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
#         self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

#         # Output layer to project back to 2D waypoints
#         self.output_proj = nn.Linear(d_model, 2)

#     def forward(
#         self,
#         track_left: torch.Tensor,
#         track_right: torch.Tensor,
#         **kwargs,
#     ) -> torch.Tensor:
#         """
#         Predicts waypoints from the left and right boundaries of the track.

#         During test time, your model will be called with
#         model(track_left=..., track_right=...), so keep the function signature as is.

#         Args:
#             track_left (torch.Tensor): shape (b, n_track, 2)
#             track_right (torch.Tensor): shape (b, n_track, 2)

#         Returns:
#             torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
#         """

#         # Concatenate the left and right track boundaries
#         x = torch.cat((track_left, track_right), dim=2)  # Shape: (b, n_track, 4)

#         # Project input to d_model
#         x = self.input_proj(x)  # Shape: (b, n_track, d_model)

#         # Prepare the query embeddings
#         query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)  # Shape: (n_waypoints, b, d_model)

#         # Transpose x to match the expected input shape for the transformer (n_track, b, d_model)
#         x = x.permute(1, 0, 2)

#         # Pass through the transformer decoder
#         tgt = self.transformer_decoder(query_embed, x)  # Shape: (n_waypoints, b, d_model)

#         # Transpose back to (b, n_waypoints, d_model)
#         tgt = tgt.permute(1, 0, 2)

#         # Project to 2D waypoints
#         waypoints = self.output_proj(tgt)  # Shape: (b, n_waypoints, 2)

#         return waypoints


#         # raise NotImplementedError


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
        nhead: int = 8,  # Add nhead parameter
        num_layers: int = 6,  # Add num_layers parameter for the number of decoder layers
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Embedding for the query (waypoints)
        self.query_embed = nn.Embedding(n_waypoints, d_model)

        # Linear layer to project input features to d_model
        self.input_proj = nn.Linear(4, d_model)  # 4 because we concatenate left and right (2+2)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output layer to project back to 2D waypoints
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """

        # Concatenate the left and right track boundaries
        x = torch.cat((track_left, track_right), dim=2)  # Shape: (b, n_track, 4)

        # Project input to d_model
        x = self.input_proj(x)  # Shape: (b, n_track, d_model)

        # Prepare the query embeddings
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, x.size(0), 1)  # Shape: (n_waypoints, b, d_model)

        # Transpose x to match the expected input shape for the transformer (n_track, b, d_model)
        x = x.permute(1, 0, 2)

        # Pass through the transformer decoder
        tgt = self.transformer_decoder(query_embed, x)  # Shape: (n_waypoints, b, d_model)

        # Transpose back to (b, n_waypoints, d_model)
        tgt = tgt.permute(1, 0, 2)

        # Project to 2D waypoints
        waypoints = self.output_proj(tgt)  # Shape: (b, n_waypoints, 2)

        return waypoints
    


# class CNNPlanner(torch.nn.Module):
#     def __init__(
#         self,
#         n_waypoints: int = 3,
#     ):
#         # super().__init__()

#         # self.n_waypoints = n_waypoints

#         # self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
#         # self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

#         super().__init__()

#         self.n_waypoints = n_waypoints

#         # Normalization parameters
#         self.register_buffer("input_mean", torch.as_tensor([0.485, 0.456, 0.406]), persistent=False)
#         self.register_buffer("input_std", torch.as_tensor([0.229, 0.224, 0.225]), persistent=False)

#         # Define the CNN backbone
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(),
#         )

#         # Define the fully connected layers
#         self.fc_layers = nn.Sequential(
#             nn.Linear(256 * 6 * 8, 512),  # Adjust the size based on the output of conv layers
#             nn.ReLU(),
#             nn.Linear(512, n_waypoints * 2)
#         )

#     def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
#         """
#         Args:
#             image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

#         Returns:
#             torch.FloatTensor: future waypoints with shape (b, n, 2)
#         """
#         # x = image
#         # x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

#         # Normalize the input image
#         x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

#         # Pass through the CNN layers
#         x = self.conv_layers(x)

#         # Flatten the output from the conv layers
#         x = x.view(x.size(0), -1)

#         # Pass through the fully connected layers
#         x = self.fc_layers(x)

#         # Reshape to (b, n_waypoints, 2)
#         waypoints = x.view(-1, self.n_waypoints, 2)

#         return waypoints

#         # raise NotImplementedError

class CNNPlanner(torch.nn.Module):
    def __init__(self, n_waypoints: int = 3):
        super().__init__()

        self.n_waypoints = n_waypoints

        # Normalization parameters
        self.register_buffer("input_mean", torch.as_tensor([0.485, 0.456, 0.406]), persistent=False)
        self.register_buffer("input_std", torch.as_tensor([0.229, 0.224, 0.225]), persistent=False)

        # Define the CNN backbone with fewer filters
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Define the fully connected layers with fewer parameters
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 6 * 8, 256),  # Adjust the size based on the output of conv layers
            nn.ReLU(),
            nn.Linear(256, n_waypoints * 2)
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        # Normalize the input image
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # Pass through the CNN layers
        x = self.conv_layers(x)

        # Flatten the output from the conv layers
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers
        x = self.fc_layers(x)

        # Reshape to (b, n_waypoints, 2)
        waypoints = x.view(-1, self.n_waypoints, 2)

        return waypoints
    
MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024



class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        # Use cross_entropy to compute the loss
        loss = F.cross_entropy(logits, target)
        return loss
