
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

def load_data(data_dir, 
                      batch_size,
                      shuffle=True,
                      num_workers=1,
                      pin_memory=False, 
                      train_mean=None,
                      train_std=None):
    # TODO
    # Define transforms from the paper Implementation section
    # Add 4-pixel-padding on each side- in paper
    # Randomly flip the padded image - in paper
    # Take a 32x32 crop of the image - in paper.
    # NOTE: we only add these transformations to the training set, 
    # not to the testing set
    # 
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    if train_mean != None and train_std != None:
        train_normalize = transforms.Normalize(mean=train_mean, std=train_std)
        
        # TODO: Add the normalizing step to train_transform
        train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Pad(padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32),
        train_normalize
    ])
        
        # TODO: Add the normalizing step to test_transform
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        train_normalize
    ])


    # Download the dataset or point to where each set is already downloaded
    train_dataset = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     download=True,
                                     transform=train_transform)
    
    test_dataset = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    download=True,
                                    transform=test_transform)
    
    # Create dataloader objects
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=shuffle,
                                               num_workers=num_workers,
                                               pin_memory=pin_memory)
    
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=pin_memory)
    
    return train_dataset, test_dataset, train_loader, test_loader