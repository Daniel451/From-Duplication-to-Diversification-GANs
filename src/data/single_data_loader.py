from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class CIFAR10SingleClass(Dataset):
    """
    A custom CIFAR-10 dataset that only includes images from a single target class.
    """

    def __init__(self, cifar10_dataset, target_class):
        self.cifar10_dataset = cifar10_dataset
        self.target_class = target_class
        self.indices = [
            i
            for i, (_, label) in enumerate(self.cifar10_dataset)
            if label == self.target_class
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.cifar10_dataset[self.indices[idx]]


def get_single_cifar10_dataloader(
    target_class, batch_size=64, num_workers=2, download_path="./data"
):
    """
    Returns CIFAR-10 data loaders for a specific class.
    Args:
    - target_class (int): The class to filter out. Integer from 0 to 9.
    - batch_size (int): Batch size.
    - num_workers (int): Number of subprocesses to use for data loading.
    - download_path (str): Directory to download CIFAR-10 dataset.

    Returns:
    - train_loader: DataLoader for the training set of the specified class.
    - test_loader: DataLoader for the test set of the specified class.
    """

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    full_train_dataset = datasets.CIFAR10(
        root=download_path, train=True, download=True, transform=transform
    )
    full_test_dataset = datasets.CIFAR10(
        root=download_path, train=False, download=True, transform=transform
    )

    train_dataset = CIFAR10SingleClass(full_train_dataset, target_class)
    test_dataset = CIFAR10SingleClass(full_test_dataset, target_class)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader, test_loader
