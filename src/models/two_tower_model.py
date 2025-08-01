import torch
import torch.nn as nn
from typing import Dict, Tuple

class UserTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class ItemTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class TwoTowerModel(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        self.user_tower = UserTower(
            config['user_input_dim'],
            config['hidden_dim'],
            config['embedding_dim']
        )
        self.item_tower = ItemTower(
            config['item_input_dim'],
            config['hidden_dim'],
            config['embedding_dim']
        )
    
    def forward(self, user_features: torch.Tensor, item_features: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_tower(user_features)
        item_emb = self.item_tower(item_features)
        return torch.cosine_similarity(user_emb, item_emb, dim=1)