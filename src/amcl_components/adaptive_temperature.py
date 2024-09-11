import torch
import torch.nn as nn

class AdaptiveTemperature(nn.Module):
    def __init__(self, input_dim, num_heads=3, temp_init=0.1, lower_limit=1e-5, upper_limit=2.0, beta=0.1):
        """
        Initialize the adaptive temperature module for multiple projection heads.
        :param input_dim: Dimensionality of the input.
        :param num_heads: Number of projection heads.
        :param temp_init: Initial temperature value.
        :param lower_limit: Lower limit of the temperature.
        :param upper_limit: Upper limit of the temperature.
        :param beta: Regularization coefficient for the temperature.
        """
        super(AdaptiveTemperature, self).__init__()
        self.num_heads = num_heads
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.beta = beta

        # Learnable temperature for each head
        self.temperature_heads = nn.ModuleList([nn.Parameter(torch.Tensor([temp_init])) for _ in range(num_heads)])

        # Shared MLP layer to compute head-wise temperature (ϕ(·)) for each projection head
        self.shared_mlp_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        ) for _ in range(num_heads)])

    def forward(self, z_i_list, z_j_list):
        """
        Forward pass to compute adaptive temperatures for each projection head.
        :param z_i_list: List of feature vectors (from anchor image) from each projection head.
        :param z_j_list: List of feature vectors (from positive or negative image) from each projection head.
        :return: List of scaled similarities and a total regularization term.
        """
        scaled_similarities = []
        total_reg_term = 0.0

        # Iterate over each head
        for head_idx in range(self.num_heads):
            z_i = z_i_list[head_idx]
            z_j = z_j_list[head_idx]

            # Compute cosine similarity for the current head
            sim = torch.cosine_similarity(z_i, z_j)

            # Compute the dot product between z_i and z_j for temperature adjustment in current head
            dot_product = self.shared_mlp_heads[head_idx](torch.cat([z_i, z_j], dim=-1))

            # Apply sigmoid to get temperature between lower_limit and upper_limit for this head
            temp = torch.sigmoid(dot_product) * (self.upper_limit - self.lower_limit) + self.lower_limit

            # Scale the similarity with the adaptive temperature
            scaled_sim = sim / temp
            scaled_similarities.append(scaled_sim)

            # Regularization term for the current head's temperature
            reg_term = self.beta * ((scaled_sim.size(1) / 2) * torch.log(temp) + 1 / temp).mean()
            total_reg_term += reg_term

        return scaled_similarities, total_reg_term
