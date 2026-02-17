import torch
import torch.nn as nn
import numpy as np
from src.generator import Generator
from src.discriminator import Discriminator


def train_gan(z_data, epochs=3000, batch_size=64):

    device = "cpu"

    G = Generator().to(device)
    D = Discriminator().to(device)

    criterion = nn.BCELoss()

    g_optimizer = torch.optim.Adam(G.parameters(), lr=0.001)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.001)

    z_data = torch.tensor(z_data, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):

        idx = np.random.randint(0, len(z_data), batch_size)
        real = z_data[idx].to(device)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # ---- Train Discriminator ----
        noise = torch.randn(batch_size, 1)
        fake = G(noise)

        d_real = D(real)
        d_fake = D(fake.detach())

        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # ---- Train Generator ----
        noise = torch.randn(batch_size, 1)
        fake = G(noise)

        g_loss = criterion(D(fake), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | D Loss {d_loss.item():.4f} | G Loss {g_loss.item():.4f}")

    return G
