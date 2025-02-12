# train.py
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchinfo import summary
from utils import save_model, train, read_data, DistilLog, analyze_data, plot_training_history

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cấu hình hyperparameters
CONFIG = {
    'num_classes': 2,
    'batch_size': 50,
    'learning_rate': 0.00001,  # Giảm learning rate
    'hidden_size': 128,
    'input_size': 30,
    'sequence_length': 10,
    'num_layers': 2,
    'dropout': 0.5,  # Tăng dropout
    'num_epochs': 300
}

# Đường dẫn files
PATHS = {
    'train_path': './Res/BGL_sequence_mapped_train.csv',
    'save_teacher_path': './Models/teacher.pth',
    'save_noKD_path': './Models/noKD.pth'
}

def main():
    # Load and preprocess data
    train_x, train_y = read_data(PATHS['train_path'], 
                                CONFIG['input_size'], 
                                CONFIG['sequence_length'])
    
    # Analyze data
    analyze_data(train_x, train_y)
    
    # Split data
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, 
        test_size=0.2, 
        stratify=train_y, 
        random_state=42
    )
    
    # Create weighted sampler
    train_weights = [1 if y == 0 else (np.sum(train_y == 0) / np.sum(train_y == 1)) 
                    for y in train_y]
    train_sampler = WeightedRandomSampler(train_weights, len(train_weights))
    
    # Create data loaders with correct data types
    train_dataset = TensorDataset(
        torch.FloatTensor(train_x).to(device),
        torch.LongTensor(train_y).to(device)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=train_sampler
    )
    
    # Initialize models
    teacher = DistilLog(
        input_size=CONFIG['input_size'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        num_classes=CONFIG['num_classes'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    noKD = DistilLog(
        input_size=CONFIG['input_size'],
        hidden_size=32,
        num_layers=1,
        num_classes=CONFIG['num_classes'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # Print model summaries
    print("\nTeacher Model Summary:")
    summary(teacher, input_size=(CONFIG['batch_size'], CONFIG['sequence_length'], CONFIG['input_size']))
    
    print("\nNoKD Model Summary:")
    summary(noKD, input_size=(CONFIG['batch_size'], CONFIG['sequence_length'], CONFIG['input_size']))
    
    # Train Teacher model
    print("\nTraining Teacher model...")
    teacher, teacher_history = train(teacher, train_loader, CONFIG['learning_rate'], CONFIG['num_epochs'])
    plot_training_history(teacher_history, 'teacher_training_history.png')
    # Train NoKD model
    print("\nTraining NoKD model...")
    noKD, noKD_history = train(noKD, train_loader, CONFIG['learning_rate'], CONFIG['num_epochs'])
    plot_training_history(noKD_history, 'noKD_training_history.png')

    
    # Save final models
    save_model(teacher, PATHS['save_teacher_path'])
    save_model(noKD, PATHS['save_noKD_path'])

if __name__ == "__main__":
    main()