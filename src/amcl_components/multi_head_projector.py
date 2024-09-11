import torch
import torch.nn as nn

class MultiHeadProjector(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads=3):
        super(MultiHeadProjector, self).__init__()
        self.num_heads = num_heads
        self.projection_heads = nn.ModuleList([nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        ) for _ in range(num_heads)])

    def forward(self, x):
        # Run the input through each projection head
        projections = [head(x) for head in self.projection_heads]
        return projections
