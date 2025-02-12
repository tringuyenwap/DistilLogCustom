import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm 
import torch.quantization
import math
import copy
from time import time 
from utils import load_data, load_model, DistilLog, save_model 

batch_size = 50
input_size = 30
sequence_length = 30
hidden_size = 128
num_layers = 2
num_classes = 2 
split = 50
device = torch.device('cpu')
save_teacher_path = './Models/teacher.pth'
save_student_path = './Models/student.pth'
save_noKD_path = './Models/noKD.pth'
test_path = './Res/BGL_sequence_mapped_test.csv'
save_quantized_path = './Models/quantized_model.pth'
output_log_path = 'output_log_test.txt'

# Thay đổi phần đầu file test.py
fi = pd.read_csv('./Res/pca_vector.csv', header=None)  # Thêm index_col=0
vec = np.array(fi.values, dtype=np.float32)  # Chỉ lấy giá trị vector, bỏ qua cột index

print(f"Shape của vector features: {vec.shape}")  # Debug thông tin
print(f"Số lượng sub-batches: {split}")

test_logs_series = pd.read_csv(test_path).values
test_total = len(test_logs_series)
sub = int(test_total / split)

def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1*n:]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r

def load_test(i):
    if i!=split-1:
        label = test_logs_series[i*sub:(i+1)*sub,1]
        logs_data = test_logs_series[i*sub:(i+1)*sub,0]
    else:
        label = test_logs_series[i*sub:,1]
        logs_data = test_logs_series[i*sub:,0]
    logs = []

    print(f"Batch {i}: Số lượng logs: {len(logs_data)}")

    for logid in range(len(logs_data)):
        ori_seq = [int(eventid) for eventid in logs_data[logid].split() if eventid.isdigit()]
        seq_pattern = mod(ori_seq, sequence_length)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append(np.ones(input_size, dtype=np.float32) * -1)
            else:
                event_idx = event - 1
                if event_idx < len(vec):
                    vec_pattern.append(vec[event_idx])
                else:
                    print(f"Warning: Event ID {event} vượt quá số lượng vectors có sẵn")
                    vec_pattern.append(np.ones(input_size, dtype=np.float32) * -1)
        logs.append(vec_pattern)

    # Chuyển đổi và kiểm tra shape
    logs = np.array(logs, dtype=np.float32)
    train_x = logs
    train_y = np.array(label, dtype=np.int64)

    print(f"Shape của train_x: {train_x.shape}")
    
    # Kiểm tra kích thước
    if train_x.shape[-1] != input_size:
        print(f"Error: Vector size không khớp. Got: {train_x.shape[-1]}, Expected: {input_size}")
        return None, None

    return train_x, train_y

def test(model, criterion=nn.CrossEntropyLoss()):
    model.eval()
    test_loss, TP, FP, FN, TN = 0, 0, 0, 0, 0
    
    with torch.no_grad():
        for i in range(split):
            test_x, test_y = load_test(i)
            if test_x is None or test_y is None or test_x.size == 0:
                print(f"Skipping batch {i} due to invalid data")
                continue

            print(f"Batch {i}: test_x shape: {test_x.shape}, test_y shape: {test_y.shape}")

            test_loader = load_data(test_x, test_y, batch_size)
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output, _ = model(data)
                test_loss += criterion(output, target).item()

                # Thay đổi cách predict
                probabilities = torch.nn.functional.softmax(output, dim=1)
                predicted = probabilities.argmax(dim=1).cpu().numpy()
                target = target.cpu().numpy()

                # Tính toán metrics
                TP += ((predicted == 1) & (target == 1)).sum()
                FP += ((predicted == 1) & (target == 0)).sum()
                FN += ((predicted == 0) & (target == 1)).sum()
                TN += ((predicted == 0) & (target == 0)).sum()

                # Debug thông tin cho batch đầu tiên
                if i == 0:
                    print("\nDebug thông tin batch đầu:")
                    print(f"Probabilities shape: {probabilities.shape}")
                    print(f"Một số probabilities đầu tiên:\n{probabilities[:5]}")
                    print(f"Predicted values: {predicted[:5]}")
                    print(f"Target values: {target[:5]}\n")

        # Tính các metrics
        P = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
        R = 100 * TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        accuracy = 100 * (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

        # Log kết quả chi tiết hơn
        with open(output_log_path, 'a') as f:
            f.write(f"\nTest set results:\n"
                   f"Confusion Matrix:\n"
                   f"TP: {TP}, FP: {FP}\n"
                   f"FN: {FN}, TN: {TN}\n"
                   f"Total samples: {TP + FP + FN + TN}\n"
                   f"Class distribution - Positive: {TP + FN}, Negative: {TN + FP}\n"
                   f"Average loss: {test_loss / (split * sub):.4f}\n"
                   f"Accuracy: {accuracy:.2f}%\n"
                   f"Precision: {P:.3f}%\n"
                   f"Recall: {R:.3f}%\n"
                   f"F1-measure: {F1:.3f}%\n")

    return accuracy, test_loss, P, R, F1, TP, FP, TN, FN

def main():
    student = DistilLog(input_size=input_size, hidden_size=4, num_layers=1, num_classes=num_classes, is_bidirectional=True).to(device)
    student = load_model(student, save_student_path)

    start_time = time()
    accuracy, test_loss, P, R, F1, TP, FP, TN, FN = test(student, criterion=nn.CrossEntropyLoss())
    test_loss /= (split * sub)
    elapsed_time = time() - start_time

    with open(output_log_path, 'a') as f:
        f.write(f"Result of testing student model\n"
                f"Total time = {elapsed_time:.2f} seconds\n")

if __name__ == "__main__":
    main()
