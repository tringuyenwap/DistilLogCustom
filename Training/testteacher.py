import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from time import time 
from utils import load_data, load_model, DistilLog, save_model 
from tqdm import tqdm

# Cấu hình device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Cấu hình hyperparameters
CONFIG = {
    'num_classes': 2,
    'batch_size': 50,
    'input_size': 30,
    'sequence_length': 10,
    'hidden_size': 128,
    'num_layers': 2,
    'is_bidirectional': False,
    'dropout': 0.3
}

# Đường dẫn files
PATHS = {
    'test_path': '../Res/BGL_sequence_mapped_test.csv',
    'save_teacher_path': '../Models/teacher.pth',
    'pca_vector_path': '../Res/pca_vector.csv',
    'output_log_path': 'output_log_test_teacher.txt'
}

# Đọc vector features từ file PCA
fi = pd.read_csv(PATHS['pca_vector_path'], header=None)
vec = np.array(fi.values, dtype=np.float32)

# Cấu hình test
split = 12  # Số lượng sub-batches để xử lý
print(f"Shape của vector features: {vec.shape}")
print(f"Số lượng sub-batches: {split}")

# Đọc file test
test_logs_series = pd.read_csv(PATHS['test_path']).values
test_total = len(test_logs_series)
sub = int(test_total / split)

def mod(l, n):
    """Truncate or pad a list"""
    r = l[-1 * n:]
    if len(r) < n:
        r.extend([0] * (n - len(r)))
    return r

def load_test(i):
    """
    Hàm load_test chia dữ liệu test thành các batch nhỏ, 
    chuyển các chuỗi log thành vector theo thứ tự đã mapping.
    """
    if i != split - 1:
        label = test_logs_series[i * sub:(i + 1) * sub, 1]
        logs_data = test_logs_series[i * sub:(i + 1) * sub, 0]
    else:
        label = test_logs_series[i * sub:, 1]
        logs_data = test_logs_series[i * sub:, 0]
    logs = []

    print(f"Batch {i}: Số lượng logs: {len(logs_data)}")

    for log_str in logs_data:
        # Chuyển chuỗi log thành danh sách số nguyên
        ori_seq = [int(eventid) for eventid in log_str.split() if eventid.isdigit()]
        seq_pattern = mod(ori_seq, CONFIG['sequence_length'])
        vec_pattern = []
        
        for event in seq_pattern:
            if event == 0:
                vec_pattern.append(np.ones(CONFIG['input_size'], dtype=np.float32) * -1)
            else:
                event_idx = event - 1
                if event_idx < len(vec):
                    vec_pattern.append(vec[event_idx])
                else:
                    print(f"Warning: Event ID {event} vượt quá số lượng vectors có sẵn")
                    vec_pattern.append(np.ones(CONFIG['input_size'], dtype=np.float32) * -1)
        logs.append(vec_pattern)

    logs = np.array(logs, dtype=np.float32)
    test_x = logs
    test_y = np.array(label, dtype=np.int64)

    print(f"Shape của test_x: {test_x.shape}")
    
    if test_x.shape[-1] != CONFIG['input_size']:
        print(f"Error: Vector size không khớp. Got: {test_x.shape[-1]}, Expected: {CONFIG['input_size']}")
        return None, None

    return test_x, test_y

def test(model, criterion=nn.CrossEntropyLoss()):
    """
    Hàm test model với tập test
    """
    model.eval()
    total_test_loss = 0.0
    total_count = 0
    TP, FP, FN, TN = 0, 0, 0, 0

    with torch.no_grad():
        for i in range(split):
            test_x, test_y = load_test(i)
            if test_x is None or test_y is None or test_x.size == 0:
                print(f"Skipping batch {i} do dữ liệu không hợp lệ")
                continue

            print(f"Batch {i}: test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")

            test_loader = load_data(test_x, test_y, CONFIG['batch_size'])
            for data, target in test_loader:
                # Chuyển data lên device
                data = data.to(device).float()
                target = target.to(device).long()
                
                # Forward pass
                output, _ = model(data)
                
                # Tính loss
                batch_loss = criterion(output, target).item() * data.size(0)
                total_test_loss += batch_loss
                total_count += data.size(0)

                # Tính các metrics
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted = probabilities.argmax(dim=1).cpu().numpy()
                target_np = target.cpu().numpy()

                # Cập nhật confusion matrix
                TP += np.sum((predicted == 1) & (target_np == 1))
                FP += np.sum((predicted == 1) & (target_np == 0))
                FN += np.sum((predicted == 0) & (target_np == 1))
                TN += np.sum((predicted == 0) & (target_np == 0))

                # Debug cho batch đầu tiên
                if i == 0:
                    print("\nDebug thông tin batch đầu:")
                    print(f"Probabilities shape: {probabilities.shape}")
                    print(f"Một số probabilities đầu tiên:\n{probabilities[:5]}")
                    print(f"Predicted values: {predicted[:5]}")
                    print(f"Target values: {target_np[:5]}\n")

    # Tính các metrics cuối cùng
    avg_loss = total_test_loss / total_count if total_count > 0 else 0.0
    P = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    R = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
    accuracy = 100 * (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Ghi kết quả vào file log
    with open(PATHS['output_log_path'], 'a') as f:
        f.write(f"\nTeacher Model Test Results:\n"
                f"Confusion Matrix:\n"
                f"TP: {TP}, FP: {FP}\n"
                f"FN: {FN}, TN: {TN}\n"
                f"Total samples: {TP + FP + FN + TN}\n"
                f"Class distribution - Positive: {TP + FN}, Negative: {TN + FP}\n"
                f"Average loss: {avg_loss:.4f}\n"
                f"Accuracy: {accuracy:.2f}%\n"
                f"Precision: {P:.3f}%\n"
                f"Recall: {R:.3f}%\n"
                f"F1-measure: {F1:.3f}%\n")
    
    # In kết quả ra console
    print(f"\nTest Results:")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {P:.3f}%")
    print(f"Recall: {R:.3f}%")
    print(f"F1-measure: {F1:.3f}%")
    
    return accuracy, avg_loss, P, R, F1, TP, FP, TN, FN

def main():
    # Khởi tạo model
    teacher = DistilLog(
        input_size=CONFIG['input_size'],
        hidden_size=CONFIG['hidden_size'],
        num_layers=CONFIG['num_layers'],
        num_classes=CONFIG['num_classes'],
        is_bidirectional=CONFIG['is_bidirectional'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    # Load model weights
    teacher = load_model(teacher, PATHS['save_teacher_path'])
    print("Teacher model loaded successfully.")
    
    # Test model
    start_time = time()
    accuracy, avg_loss, P, R, F1, TP, FP, TN, FN = test(
        teacher, 
        criterion=nn.CrossEntropyLoss()
    )
    elapsed_time = time() - start_time

    # Ghi thời gian test
    with open(PATHS['output_log_path'], 'a') as f:
        f.write(f"Total testing time = {elapsed_time:.2f} seconds\n")
    print(f"Total testing time = {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()