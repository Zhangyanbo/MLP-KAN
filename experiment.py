import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from mlp_kan import KANLayer
from tqdm import tqdm


def target_fn(input):
    # f(x,y)=exp(sin(pi * x)+y^2)
    if len(input.shape) == 1:
        x, y = input
    else:
        x, y = input[:, 0], input[:, 1]
    # return torch.exp(torch.sin(torch.pi * x) + y**2)
    return x * y


def train(model, n_epoch, lr=1e-2, batch_size=32, weight_decay=1e-3, L1=1e-5, eta_min=1e-5):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epoch, eta_min=eta_min)
    lf = nn.MSELoss()

    loss_count = []

    for i in tqdm(range(n_epoch)):
        optimizer.zero_grad()
        x = torch.rand(batch_size, 2) #* 2 - 1
        y = target_fn(x)
        y_pred = model(x)
        loss = lf(y_pred.reshape(-1), y)
        L1_loss = 0
        for param in model.parameters():
            L1_loss += torch.norm(param, 1)
        train_loss = loss + L1 * L1_loss
        train_loss.backward()
        loss_count.append(loss.item())
        optimizer.step()
        scheduler.step()
    
    return loss_count


if __name__ == '__main__':

    dims = [2, 1, 1]
    model = nn.Sequential(
        KANLayer(dims[0], dims[1]),
        KANLayer(dims[1], dims[2])
    )

    loss_count = train(model, 10000, lr=1e-3, weight_decay=1e-4, L1=0)

    plt.plot(loss_count)
    plt.semilogy()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.close()

    layer = 0

    plt.figure(figsize=(3*dims[1], 2*dims[0]))
    x_ranges = [[0, 0]] * dims[1] # saving the y range as the x range to the second layer
    for i in range(dims[0]):
        x_min, x_max = 0, 0
        for j in range(dims[1]):
            plt.subplot(dims[0], dims[1], i * dims[1] + j + 1)
            f = model[layer].take_function(i, j)

            x = torch.linspace(0, 1, 100)
            y = f(x.unsqueeze(-1)).detach().squeeze()

            vmin, vmax = y.min().item(), y.max().item()
            x_ranges[j] = [x_ranges[j][0] + vmin, x_ranges[j][1] + vmax]

            plt.plot(x, y)
            plt.ylim(vmin - 0.1, vmax + 0.1)
            plt.title(f"$f_{{{i}, {j}}}$")

    plt.tight_layout()
    plt.savefig('./temp/layer_0_inter.png')
    plt.close()

    layer = 1

    plt.figure(figsize=(3*dims[1], 2*dims[2]))
    for i in range(dims[1]):
        for j in range(dims[2]):
            plt.subplot(dims[2], dims[1], i * dims[2] + j + 1)
            f = model[layer].take_function(i, j)

            x = torch.linspace(x_ranges[i][0] - 0.05, x_ranges[i][1] + 0.05, 100)
            y = f(x.unsqueeze(-1)).detach().squeeze()

            vmin, vmax = y.min().item() - 0.1, y.max().item() + 0.1

            plt.plot(x, y)
            plt.ylim(vmin, vmax)
            plt.xlim(x_ranges[i][0] - 0.05, x_ranges[i][1] + 0.05)
            plt.title(f"$f_{{{i}, {j}}}$")

    plt.tight_layout()
    plt.savefig('./temp/layer_1_inter.png')
    plt.close()
