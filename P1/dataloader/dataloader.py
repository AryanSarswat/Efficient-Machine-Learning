import torch
from torchvision import datasets, transforms

def load_cifar10():
    data_dir = '../data'
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
        
    train_set = datasets.CIFAR10('../data', train=True, download=True,
                    transform=transform)
    test_set = datasets.CIFAR10('../data', train=False,
                    transform=transform)
    
    return train_set, test_set

def load_mnist():
    data_dir = '../data'
    
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
        
    train_set = datasets.MNIST('../data', train=True, download=True,
                    transform=transform)
    test_set = datasets.MNIST('../data', train=False,
                    transform=transform)
    
    return train_set, test_set
    
def load_data(config):
    if config['dataset'] == 'cifar10':
        train_set, test_set = load_cifar10()
    elif config['dataset'] == 'mnist':
        train_set, test_set = load_mnist()
    else:
        raise ValueError('Invalid dataset')
    
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               batch_size=config['batch_size'],
                                               num_workers=config['num_workers'],
                                               pin_memory=config['pin_memory'],
                                               shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=config['batch_size'],
                                              num_workers=config['num_workers'],
                                              pin_memory=config['pin_memory'],
                                              shuffle=False)
    
    return train_loader, test_loader
    
if __name__ == "__main__":
    config = {'dataset': 'mnist',
                'batch_size': 64,
                'num_workers': 4,
                'pin_memory': True if torch.cuda.is_available() else False,
                }
    
    train_loader, test_loader = load_data(config)