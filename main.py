import yaml
import torch
from src.models.two_tower_model import TwoTowerModel
from src.pipeline.data_pipeline import DataPipeline
from src.training.trainer import TwoTowerTrainer
from src.narrative.llm_judge import LLMJudge
from src.inference.recommendation_engine import RecommendationEngine
import wandb

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize wandb
    wandb.init(project="two-tower-recommendation")
    
    # Setup data pipeline
    data_pipeline = DataPipeline()
    
    # Initialize model
    model_config = {
        'user_input_dim': 10,
        'item_input_dim': 8,
        'hidden_dim': config['model']['hidden_dim'],
        'embedding_dim': config['model']['embedding_dim']
    }
    model = TwoTowerModel(model_config)
    
    # Initialize trainer
    trainer = TwoTowerTrainer(model, config['training'])
    
    # Initialize LLM judge
    llm_judge = LLMJudge(config['llm_judge'])
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine(model, llm_judge)
    
    print("Production pipeline initialized successfully!")

if __name__ == "__main__":
    main()