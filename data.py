import os 
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class TorDataSet(Dataset):
    def __init__(self, file_path, maxlen=None, minlen=0, traces=0, val_size=0.1, test_size=0.1):
        self.file_path = file_path
        self.maxlen = maxlen
        self.minlen = minlen
        self.traces = traces
        self.val_size = val_size
        self.test_size = test_size

        # Load data and preprocess data
        self.data, self.label, self.dict_labels = self.load_and_preprocess_data()

        # Split data into train, val, and test sets
        (self.data_train, self.label_train), (self.data_val, self.label_val), (self.data_test, self.label_test) = self.split_data(
            self.data, self.label, val_split=self.val_size, test_split=self.test_size
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.label[idx], dtype=torch.long)
    
    def get_trainset(self):
        return torch.utils.data.TensorDataset(torch.tensor(self.data_train, dtype=torch.float32), torch.tensor(self.label_train, dtype=torch.long))

    def get_valset(self):
        return torch.utils.data.TensorDataset(torch.tensor(self.data_val, dtype=torch.float32), torch.tensor(self.label_val, dtype=torch.long))

    def get_testset(self):
        return torch.utils.data.TensorDataset(torch.tensor(self.data_test, dtype=torch.float32), torch.tensor(self.label_test, dtype=torch.long))

    def categorize(self, labels):
        unique_labels = sorted(list(set(labels)))
        dict_labels = {label: i for i, label in enumerate(unique_labels)}
        numerical_labels = [dict_labels[label] for label in labels]
        return numerical_labels, dict_labels
    
    def load_and_preprocess_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f'{self.file_path} not found')
        
        # Load the data from the file
        with np.load(self.file_path, allow_pickle=True) as f:
            data = f['data']
            labels = f['labels']

        # Data filtering base on length and traces
        if self.minlen > 0 or self.traces > 0:
            print(f"Filtering data with length >= {self.minlen} and <= {self.maxlen} and traces <= {self.traces}")
            filtered_data, filtered_labels = [], []
            num_traces = {}
            for x, y in zip(data, labels):
                if y not in num_traces:
                    num_traces[y] = 0
                
                if self.traces > 0 and num_traces[y] >= self.traces:
                    continue
                if len(x) >= self.minlen:
                    filtered_data.append(x)
                    filtered_labels.append(y)
                    num_traces[y] += 1
                
            data = np.array(filtered_data)
            labels = np.array(filtered_labels)
            if data.size == 0:
                raise ValueError("No data left after filtering")
        
        if self.maxlen is not None:
            print(f"Truncating/padding data to length {self.maxlen}")
            data = np.array([x[:self.maxlen] if len(x) > self.maxlen else np.pad(x, (0, self.maxlen - len(x)), 'constant') for x in data])
        
        data = np.expand_dims(data, axis=-1).astype(np.float32)
        numerical_labels, dict_labels = self.categorize(labels)
        labels = np.array(numerical_labels, dtype=np.int32)
        print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
        return data, labels, dict_labels

    def split_data(self, data, labels, val_split=0.1, test_split=0.1):
        # First split into train+val and test sets
        data_train_val, data_test, labels_train_val, labels_test = train_test_split(
            data, labels, test_size=test_split, stratify=labels, random_state=42
        )

        # Then split train+val into train and val
        val_size = val_split / (1 - test_split)  # Adjust val split proportion
        data_train, data_val, labels_train, labels_val = train_test_split(
            data_train_val, labels_train_val, test_size=val_size, stratify=labels_train_val, random_state=42
        )
        print(f"Train size: {len(data_train)}, Val size: {len(data_val)}, Test size: {len(data_test)}")
        return (data_train, labels_train), (data_val, labels_val), (data_test, labels_test)
        