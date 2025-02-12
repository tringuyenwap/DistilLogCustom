# ultils.py (hoặc utils.py)
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.modules.module import Module
from torchinfo import summary
from tqdm import tqdm
import csv
from time import time
from torch.nn import functional as F
from attention_layers import LinearAttention
import torch.nn.init as init
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DistilLog(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, is_bidirectional=False,
                 dropout=0.3):
        super(DistilLog, self).__init__()
        self.num_directions = 2 if is_bidirectional else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        # Input processing
        self.batch_norm = nn.BatchNorm1d(input_size)
        self.input_dropout = nn.Dropout(0.2)
        
        # GRU layer
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=is_bidirectional
        )
        
        # Attention mechanism
        self.attention_size = hidden_size * self.num_directions
        self.w_omega = nn.Parameter(
            torch.randn(hidden_size * self.num_directions, self.attention_size)
        )
        self.u_omega = nn.Parameter(torch.randn(self.attention_size))
        
        # Output layers
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * self.num_directions, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'w_omega' in name or 'u_omega' in name:
                nn.init.normal_(param)
                
    def attention(self, gru_output, seq_len):
        # Reshape for attention
        output_reshape = torch.Tensor.reshape(gru_output,
                                          [-1, self.hidden_size * self.num_directions])
        
        # Attention mechanism
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega.to(device)))
        attn_hidden_layer = torch.mm(
            attn_tanh, torch.Tensor.reshape(self.u_omega.to(device), [-1, 1]))
        
        # Calculate attention weights
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer),
                                [-1, seq_len])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas,
                                          [-1, seq_len, 1])
        
        # Apply attention weights
        state = gru_output
        attn_output = torch.sum(state * alphas_reshape, 1)
        return attn_output, alphas

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        
        # Apply batch normalization
        x_reshaped = x.reshape(-1, x.shape[-1])
        x_normalized = self.batch_norm(x_reshaped)
        x = x_normalized.reshape(batch_size, sequence_length, -1)
        
        # Apply input dropout
        x = self.input_dropout(x)
        
        # GRU layer
        gru_output, _ = self.gru(x)
        
        # Apply attention
        attn_output, attention_weights = self.attention(gru_output, sequence_length)
        
        # Apply layer normalization
        normalized_output = self.layer_norm(attn_output)
        
        # Fully connected layers with dropout
        x = self.dropout(normalized_output)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits, attention_weights

def load_model(model, save_path):
    model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
    return model

def save_model(model, save_path):
    torch.save(model.state_dict(), save_path)

def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1 * n:]
    if len(r) < n:
        r.extend([0] * (n - len(r)))
    return r

def analyze_data(train_x, train_y):
    print("\n=== PHÂN TÍCH DỮ LIỆU ===")
    print(f"Số lượng mẫu: {len(train_x)}")
    print(f"Phân bố nhãn:\n{pd.Series(train_y).value_counts()}")
    
    print("\nKiểm tra giá trị thiếu:")
    print(f"Missing values trong train_x: {np.isnan(train_x).sum()}")
    
    print("\nThống kê features:")
    print(f"Mean: {np.mean(train_x)}")
    print(f"Std: {np.std(train_x)}")
    print(f"Min: {np.min(train_x)}")
    print(f"Max: {np.max(train_x)}")


def read_data(path, input_size, sequence_length):
    fi = pd.read_csv('../Res/pca_vector.csv', header = None)
    vec = []
    vec = fi
    vec = np.array(vec)

    logs_series = pd.read_csv(path)
    logs_series = logs_series.values
    label = logs_series[:, 1]
    logs_data = logs_series[:, 0]
    logs = []
    for i in range(0, len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[i].split()]
        seq_pattern = mod(ori_seq, sequence_length)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append([-1] * input_size)
            else:
                vec_pattern.append(vec[event - 1])
        logs.append(vec_pattern)
    logs = np.array(logs)
    train_x = logs
    train_y = np.array(label)
    train_x = np.reshape(train_x, (train_x.shape[0], -1, input_size))
    train_y = train_y.astype(int)

    return train_x, train_y

def load_data(train_x, train_y, batch_size):
    tensor_x = torch.Tensor(train_x)
    tensor_y = torch.LongTensor(train_y)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    return train_loader

def train(model, train_loader, learning_rate, num_epochs):
    # Class weights
    train_labels = torch.tensor([y.item() for _, y in train_loader.dataset])
    num_pos = (train_labels == 1).sum()
    num_neg = (train_labels == 0).sum()
    pos_weight = num_neg / num_pos
    class_weights = torch.FloatTensor([1.0, pos_weight]).to(device)
    
    # Loss và optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(  # Đổi sang AdamW để có weight decay tốt hơn
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01  # L2 regularization
    )
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        patience=5, 
        factor=0.1,
        min_lr=1e-6
    )
    
    # Training history
    history = {
        'train_losses': [],
        'train_accs': []
    }
    
    # Theo dõi loss tốt nhất cho early stopping
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            # Đảm bảo kiểu dữ liệu đúng
            data = data.to(device).float()
            target = target.to(device).long()
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output, _ = model(data)
            
            # Tính loss và check NaN
            loss = criterion(output, target)
            if torch.isnan(loss):
                print("NaN loss detected! Skipping batch...")
                continue
            
            # Gradient clipping trước backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping sau backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Check gradients for NaN
            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print("NaN gradients detected! Skipping batch...")
                optimizer.zero_grad()
                continue
            
            # Optimizer step
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            if (batch_idx + 1) % 10 == 0:
                acc = 100. * correct / total if total > 0 else 0
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_description(
                    f'Epoch: {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100. * correct / total if total > 0 else 0
        
        # Update history
        history['train_losses'].append(epoch_loss)
        history['train_accs'].append(epoch_acc)
        
        # Update learning rate
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            save_model(model, f'best_model_epoch_{epoch}.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping tại epoch {epoch}")
                break
        
        # Print epoch results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return model, history
def plot_training_history(history, save_path='training_history.png'):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot losses
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    model = DistilLog(input_size=30, hidden_size=128, num_layers=2, num_classes=2, is_bidirectional=False)
    inp = torch.rand((1, 8, 30))
    print("Input shape:", inp.shape)
    out, _ = model(inp)
    print("Output shape:", out.shape)
