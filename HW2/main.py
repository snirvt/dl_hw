import torch
import torchvision.datasets as datasets


mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)


train_loader = torch.utils.data.DataLoader(mnist_trainset,
                            batch_size=128,
                            shuffle=True,
                            num_workers=4)

test_loader = torch.utils.data.DataLoader(mnist_testset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=4)