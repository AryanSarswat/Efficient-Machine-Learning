import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader.dataloader import load_data
from models.basic_model import BasicNet
from models.resnet18 import ResNet18
from models.resnet18_modified import ResNet18_Modified
from ptflops import get_model_complexity_info
from torch.optim.lr_scheduler import CyclicLR, ExponentialLR, ReduceLROnPlateau


class EarlyStopper:
    """
    _summary_ : Early stopping class
    _params_ : 
        patience : Number of epochs to wait before stopping
        min_delta : Minimum change in the monitored quantity to qualify as an improvement
        verbose : If True, prints a message for each epoch where the loss decreases
    """
    def __init__(self, patience=5, min_delta=0.025, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')
    
    def __call__(self, loss):
        if loss < self.min_loss:
            if self.verbose:
                print(f"[INFO]: Loss decreased from {self.min_loss:.4f} to {loss:.4f}")
            self.min_loss = loss
            self.counter = 0
        elif loss > self.min_loss - self.min_delta:
            self.counter += 1
            if self.counter > self.patience:
                return True
            if self.verbose:
                print(f"[INFO]: Loss increased/stayed the same from {self.min_loss:.4f} to {loss:.4f}")
        return False
    
def measure_inference_time(model, device, input_shape=(1, 28, 28), repetitions=1000):
    """
    _summary_ : Measure inference time of a model on a dummy input

    Args:
        model (torch.nn.Module): Model to measure inference time of
        device (torch.device): Device to run the model on
        input_shape (tuple, optional): Shape of input for the model. Defaults to (1, 28, 28).
        repetitions (int, optional): Number of times to measure the time. Defaults to 1000.

    Returns:
        tuple: (mean_time, std_time)
    """
    model.to(device)
    dummy_input = torch.randn((1, *input_shape), dtype=torch.float).to(device)
    
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions,1))
    
    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)
        
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    
    return mean_syn, std_syn

def measure_flops(model, input_shape=(1, 28, 28)):
    """
    _summary_ : Measure flops and parameters of a model

    Args:
        model (torch.nn.Module): Model to measure flops and parameters of
        input_shape (tuple, optional): Shape of input for the model. Defaults to (1, 28, 28).

    Returns:
        (tuple): flops, params
    """
    flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False, verbose=False)
    flops = flops / 1e6
    params = params / 1e6
    return flops, params

def train_epoch(model, data_loader, optimizer, criterion, device):
    """
    _summary_ : Run one training epoch

    Args:
        model (torch.nn.Module): Model to train
        data_loader (torch.data.DataLoader): Training data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run the model on

    Returns:
        (tuple): train_loss, train_accuracy
    """
    model.train()
    
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    
    for i, data in enumerate(data_loader):
        counter += 1
        
        inputs, targets = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        train_running_loss += loss.item()
            
        preds = torch.argmax(outputs, dim=1)
        
        train_running_correct += (preds == targets).sum().item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
    
    train_loss = train_running_loss / counter
    train_accuracy = 100. * train_running_correct / len(data_loader.dataset)
    return train_loss, train_accuracy

def validate(model, data_loader, criterion, device):
    """
    _summary_ : Run one validation epoch

    Args:
        model (torch.nn.Module): Model to validate
        data_loader (torch.data.DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        device (torch.device): Device to run the model on

    Returns:
        (tuple): val_loss, val_accuracy
    """
    model.eval()
    
    val_running_loss = 0.0
    val_running_correct = 0
    counter = 0
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            counter += 1
            inputs, targets = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            val_running_correct += (preds == targets).sum().item()
    
    val_loss = val_running_loss / counter
    val_accuracy = 100. * val_running_correct / len(data_loader.dataset)
    return val_loss, val_accuracy

def train_validate(config):
    """
    _summary_ : Train and validate a model

    Args:
        config (dict): Configuration dictionary

    Returns:
        (tuple): val_loss_array, val_accuracy_array, train_loss_array, train_accuracy_array, flops, params, inference_time
    """
    early_stopper = EarlyStopper()
    train_loader, test_loader = load_data(config)
    device = torch.device(config['device'])
    
    if config['model'] == 'basic':
        model = BasicNet(config).to(device)
    elif config['model'] == 'resnet18':
        model = ResNet18(config).to(device)
    elif config['model'] == 'resnet18_modified':
        model = ResNet18_Modified(config).to(device)
    else:
        raise ValueError('Invalid model')
    
    criterion = config['criterion']()
    if config['optimizer'] == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=config['lr'])
    else:
        raise ValueError('Invalid optimizer')
    
    if config['lr_scheduler'] == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=1, min_lr=0, eps=1e-06, verbose=True)
    elif config['lr_scheduler'] == "ExponentialLR":
        scheduler = ExponentialLR(optimizer, gamma=0.85, last_epoch=-1, verbose=True)
    elif config['lr_scheduler'] == "CycleLR":
        scheduler = CyclicLR(optimizer, base_lr=0.001, max_lr=0.1, step_size_up=5, step_size_down=None, mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1, verbose=True)

        
    
    val_loss_array = []
    val_accuracy_array = []
    train_loss_array = []
    train_accuracy_array = []
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"[INFO]: Epoch {epoch}/{config['epochs']}")
        
        train_epoch_loss, train_epoch_accuracy = train_epoch(model, train_loader, optimizer, criterion, device)
        val_epoch_loss, val_epoch_accuracy = validate(model, test_loader, criterion, device)
        
        print(f"Train loss: {train_epoch_loss:.4f}, Train accuracy: {train_epoch_accuracy:.2f}")
        print(f"Validation loss: {val_epoch_loss:.4f}, Validation accuracy: {val_epoch_accuracy:.2f}")
        print('-' * 50)
        
        train_loss_array.append(train_epoch_loss)
        train_accuracy_array.append(train_epoch_accuracy)
        val_loss_array.append(val_epoch_loss)
        val_accuracy_array.append(val_epoch_accuracy)
        
        early_stop = early_stopper(val_epoch_loss)
        scheduler.step(val_epoch_loss)
        
        if early_stop:
            print("[INFO]: Early stopping")
            break
        
    flops, params = measure_flops(model, input_shape=(config['input_channels'], 28, 28) if config['dataset'] == 'mnist' else (config['input_channels'], 32, 32))
    inference_time = measure_inference_time(model, device, input_shape=(config['input_channels'], 28, 28) if config['dataset'] == 'mnist' else (config['input_channels'], 32, 32))
    
    return val_loss_array, val_accuracy_array, train_loss_array, train_accuracy_array, flops, params, inference_time

    

if __name__ == "__main__":
    config = {
        'model' : 'resnet18_modified',
        'dataset' : 'cifar10',
        'batch_size' : 256,
        'num_workers' : 2,
        'lr' : 1e-3,
        'epochs' : 0,
        'criterion' : nn.CrossEntropyLoss,
        'optimizer' : 'Adam',
        'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
        'input_channels' : 3,
        'num_classes' : 10,
        'pin_memory' : True if torch.cuda.is_available() else False,
        'weight_decay' : 1e-5,
        'depthwise_convs' : [False, False, False, False],
        'lr_scheduler' : 'ReduceLROnPlateau'
    }
    
    depthwise_configs = [
        [False, False, False, False],
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
        [False, False, False, True],
        [False, False, True, True],
        [False, True, True, True],
    ]

    
    for depthwise_config in depthwise_configs:
        config['depthwise_convs'] = depthwise_config
        print(train_validate(config))
    
    
    