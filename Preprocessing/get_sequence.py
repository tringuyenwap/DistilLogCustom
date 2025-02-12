import os
import pandas as pd
import numpy as np

# Define parameters
para = {
    "window_size": 20,  # window size in log entries
    "step_size": 20,    # step size in log entries
    "structured_file": "../Res/IoTDB1.log_strucctured.csv",
    "BGL_sequence": '../Res/BGL_sequence.csv'
}

def load_BGL():
    structured_file = para["structured_file"]
    # Load data
    bgl_structured = pd.read_csv(structured_file)
    
    # Get the label for each log (1 for abnormal, 0 for normal)
    bgl_structured['label'] = (bgl_structured['Label'] != '-').astype(int)
    
    return bgl_structured

def bgl_sampling(bgl_structured):
    label_data = bgl_structured['label'].values
    event_mapping_data = bgl_structured['EventId'].values
    log_size = len(label_data)
    
    # Calculate number of windows
    n_windows = (log_size - para["window_size"]) // para["step_size"] + 1
    
    # Initialize lists to store sequences and labels
    expanded_event_list = []
    labels = []
    
    # Create fixed-size windows
    for i in range(n_windows):
        start_idx = i * para["step_size"]
        end_idx = start_idx + para["window_size"]
        
        # Get events in current window
        window_events = event_mapping_data[start_idx:end_idx].tolist()
        expanded_event_list.append(window_events)
        
        # If any log in window is anomaly, mark window as anomaly
        window_label = 1 if any(label_data[start_idx:end_idx]) else 0
        labels.append(window_label)
    
    print(f'There are {n_windows} instances (windows) in this dataset')
    print(f"Among all instances, {sum(labels)} are anomalies")
    
    # Save sequence and labels
    BGL_sequence = pd.DataFrame({
        'sequence': expanded_event_list,
        'label': labels
    })
    BGL_sequence.to_csv(para["BGL_sequence"], index=None)

if __name__ == "__main__":
    bgl_structured = load_BGL()
    bgl_sampling(bgl_structured)