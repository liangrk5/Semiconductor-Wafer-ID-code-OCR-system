import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,ConcatDataset
import numpy as np
from data import *
from model import *
from tqdm import tqdm
import json

def run_optuna_study(dataset, n_trials=50, epochs=30, timeout=36000):
    # 获取数据集基本信息
    num_classes = len(dataset.dict_labels)
    max_len = dataset.data.shape[1]
    
    # 定义目标函数
    def objective(trial):
        # 超参数采样
        params = {
            'branch_channels': [
                trial.suggest_int("branch1_ch", 16, 128, step=16),
                trial.suggest_int("branch2_ch", 16, 128, step=16),
                trial.suggest_int("branch3_ch", 16, 128, step=16)
            ],
            'branch_kernel_sizes': [
                trial.suggest_categorical("branch1_ksize", [3,5,7]),
                trial.suggest_categorical("branch2_ksize", [3,5,7]),
                trial.suggest_categorical("branch3_ksize", [3,5,7])
            ],
            'branch_pools': [
                trial.suggest_categorical("branch1_pool", [True, False]),
                trial.suggest_categorical("branch2_pool", [True, False]),
                trial.suggest_categorical("branch3_pool", [True, False])
            ],
            'transformer_layers': trial.suggest_int("transformer_layers", 1, 3),
            'transformer_ff_dim': trial.suggest_categorical("transformer_ff", [256, 512, 1024]),
            'attention_heads': trial.suggest_categorical("attn_heads", [2,4,8]),
            'activation': trial.suggest_categorical("activation", ['relu', 'leaky_relu', 'selu']),
            'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop"]),
            'lr': trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical("batch_size", [32, 64, 128]),
            'classifier_layers': trial.suggest_int("cls_layers", 0, 2),
            'classifier_dropout': trial.suggest_float("cls_dropout", 0.0, 0.5),
            'transformer_dropout': trial.suggest_float("trans_dropout", 0.0, 0.3),
        }
        
        # 构建分类器隐藏层
        classifier_hidden = []
        for i in range(params['classifier_layers']):
            dim = trial.suggest_int(f"cls_dim_{i}", 128, 512, step=128)
            classifier_hidden.append(dim)
        
        # 实例化模型
        model = SACNN(
            num_classes=num_classes,
            max_len=max_len,
            attention_heads=params['attention_heads'],
            activation=params['activation'],
            branch_channels=params['branch_channels'],
            branch_kernel_sizes=params['branch_kernel_sizes'],
            branch_pools=params['branch_pools'],
            transformer_layers=params['transformer_layers'],
            transformer_dim_feedforward=params['transformer_ff_dim'],
            transformer_dropout=params['transformer_dropout'],
            classifier_hidden_dims=classifier_hidden,
            classifier_dropout=params['classifier_dropout'],
            use_positional_encoding=True
        )
        
        # 设备设置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # 优化器和损失函数
        optimizer = getattr(torch.optim, params['optimizer'])(
            model.parameters(), lr=params['lr'])
        criterion = nn.CrossEntropyLoss()
        
        # 数据加载器
        train_loader = DataLoader(
            dataset.get_trainset(), 
            batch_size=params['batch_size'], 
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset.get_valset(),
            batch_size=params['batch_size'],
            shuffle=False,
            pin_memory=True
        )
        
        # 训练参数
        best_val_acc = 0.0
        early_stop = 0
        patience = 5
        
        for epoch in tqdm(range(epochs)):
            # 训练阶段
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # 验证阶段
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_acc = correct / total
            trial.report(val_acc, epoch)
            
            # 早停和剪枝逻辑
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                early_stop = 0
            else:
                early_stop += 1
                
            if early_stop >= patience:
                break
                
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return best_val_acc

    def train_with_best_params(best_params, full_train=True):
        # 合并训练集和验证集
        full_trainset = ConcatDataset([
            dataset.get_trainset(),
            dataset.get_valset()
        ]) if full_train else dataset.get_trainset()

        # 构建最终模型
        classifier_hidden = []
        for i in range(best_params['cls_layers']):
            classifier_hidden.append(best_params[f'cls_dim_{i}'])

        final_model = SACNN(
            num_classes=len(dataset.dict_labels),
            max_len=dataset.data.shape[1],
            attention_heads=best_params['attn_heads'],
            activation=best_params['activation'],
            branch_channels=[
                best_params['branch1_ch'],
                best_params['branch2_ch'],
                best_params['branch3_ch']
            ],
            branch_kernel_sizes=[
                best_params['branch1_ksize'],
                best_params['branch2_ksize'],
                best_params['branch3_ksize']
            ],
            branch_pools=[
                best_params['branch1_pool'],
                best_params['branch2_pool'],
                best_params['branch3_pool']
            ],
            transformer_layers=best_params['transformer_layers'],
            transformer_dim_feedforward=best_params['transformer_ff'],
            transformer_dropout=best_params['trans_dropout'],
            classifier_hidden_dims=classifier_hidden,
            classifier_dropout=best_params['cls_dropout']
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        final_model.to(device)

        # 优化器设置
        optimizer = getattr(torch.optim, best_params['optimizer'])(
            final_model.parameters(), lr=best_params['lr']
        )
        criterion = nn.CrossEntropyLoss()

        # 数据加载器
        train_loader = DataLoader(
            full_trainset,
            batch_size=best_params['batch_size'],
            shuffle=True,
            pin_memory=True,
            num_workers=4
        )
        test_loader = DataLoader(
            dataset.get_testset(),
            batch_size=best_params['batch_size'],
            shuffle=False,
            pin_memory=True
        )

        # 训练进度监控
        best_acc = 0.0
        history = {'train_loss': [], 'val_acc': []}
        early_stop = 0
        patience = 7

        # 使用更多epoch进行最终训练
        for epoch in range(int(epochs * 1.5)):  # 延长50%训练时间
            # 训练阶段带进度条
            final_model.train()
            train_loss = 0.0
            with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}") as pbar:
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = final_model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * inputs.size(0)
                    pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # 计算平均训练损失
            avg_train_loss = train_loss / len(train_loader.dataset)
            history['train_loss'].append(avg_train_loss)

            # 验证阶段（使用测试集作为最终验证）
            final_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = final_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_acc = correct / total
            history['val_acc'].append(val_acc)

            # 打印实时指标
            print(f"Epoch {epoch+1}: "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Test Acc: {val_acc:.4f}")

            # 早停和保存最佳模型
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(final_model.state_dict(), 'best_model.pth')
                early_stop = 0
            else:
                early_stop += 1

            if early_stop >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 加载最佳模型
        final_model.load_state_dict(torch.load('best_model.pth'))
        
        # 最终测试评估
        final_model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = final_model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算评估指标
        from sklearn.metrics import classification_report
        print("\nFinal Evaluation:")
        print(classification_report(all_labels, all_preds, 
                                   target_names=dataset.dict_labels.keys()))

        return final_model, history

    # 创建Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=10)
    )
    
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    with open("best_params.json", "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2, ensure_ascii=False)
    print("✅ Best parameters saved to best_params.json")
    
    print("\nStarting final training with best parameters...")
    best_params = study.best_params
    final_model, history = train_with_best_params(best_params)
    
    return study, final_model, history

# 使用示例
if __name__ == "__main__":
    # 初始化数据集
    dataset = TorDataSet(
        file_path='tor_100w_2500tr.npz',
        maxlen=3000,
        minlen=0,
        traces=2500,
        val_size=0.1,
        test_size=0.1
    )
    
    # 运行Optuna优化
    study = run_optuna_study(dataset, n_trials=50, epochs=30)