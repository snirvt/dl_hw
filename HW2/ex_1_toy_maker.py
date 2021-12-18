
import numpy as np
import torch

def get_toy_data(n = 10000, T = 50):
    data = torch.rand((10000,50))

    for i in range(10000):
        idx = np.random.randint(20,31)
        data[i,idx-5:idx+5] *= 0.1
    return data


def toy_data_splitter(data, batch_size):
    train_loader = torch.utils.data.DataLoader(data[:6000],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1)

    val_loader = torch.utils.data.DataLoader(data[6000:8000],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1)

    test_loader = torch.utils.data.DataLoader(data[8000:],
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=1)
    return train_loader, val_loader, test_loader











