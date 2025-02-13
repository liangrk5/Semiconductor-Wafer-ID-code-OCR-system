import numpy as np
import torch
from torch.utils.data import DataLoader
from data import TorDataSet  



def main():
    file_path = 'tor_100w_2500tr.npz'

    maxlen = 200
    minlen = 0
    traces_per_label = 2500
    
    # Initialize the dataset
    dataset = TorDataSet(file_path=file_path, maxlen=maxlen, minlen=minlen, traces=traces_per_label, val_size=0.1, test_size=0.1)
    
    # Check the data shape and labels
    print(f"Data shape: {dataset.data.shape}, Labels shape: {dataset.label.shape}")
    print(f"Label Dictionary: {dataset.dict_labels}")
    
    # Retrieve train, val, and test sets
    trainset = dataset.get_trainset()
    valset = dataset.get_valset()
    testset = dataset.get_testset()

    print(f"Training set size: {len(trainset)}")
    print(f"Validation set size: {len(valset)}")
    print(f"Test set size: {len(testset)}")

    # Optionally, use DataLoader to iterate over batches
    train_loader = DataLoader(trainset, batch_size=10, shuffle=True)
    for data, labels in train_loader:
        print(f"Batch data shape: {data.shape}, Batch labels shape: {labels.shape}")
        break 


if __name__ == "__main__":
    main()
