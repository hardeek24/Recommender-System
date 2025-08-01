import pandas as pd
from typing import Tuple, Dict
from sklearn.preprocessing import StandardScaler
import torch

class DataPipeline:
    def __init__(self):
        self.user_scaler = StandardScaler()
        self.item_scaler = StandardScaler()
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Extract features
        user_features = self.user_scaler.fit_transform(data[['user_age', 'user_gender', 'user_location']])
        item_features = self.item_scaler.fit_transform(data[['item_category', 'item_price', 'item_rating']])
        labels = data['interaction'].values
        
        return (
            torch.FloatTensor(user_features),
            torch.FloatTensor(item_features),
            torch.FloatTensor(labels)
        )
    
    def create_batches(self, user_features: torch.Tensor, item_features: torch.Tensor, 
                      labels: torch.Tensor, batch_size: int):
        dataset = torch.utils.data.TensorDataset(user_features, item_features, labels)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)