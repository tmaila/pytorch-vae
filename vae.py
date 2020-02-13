import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt


class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):
    """
    Encoder for the variational autoencoder, encodes from the input space to the latent space.
    Forward pass takes an image vector and returns a tuple (mu, log(sigma)).
    """

    def __init__(self, input_size, hidden1_size, hidden2_size, latent_size):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.latent_size = latent_size

        self._hidden1 = torch.nn.Linear(input_size, hidden1_size)
        self._hidden2 = torch.nn.Linear(hidden1_size, hidden2_size)
        self._latnet_mu = torch.nn.Linear(hidden2_size, latent_size)
        self._latent_sigma = torch.nn.Linear(hidden2_size, latent_size)

    def forward(self, x):
        h1 = F.relu(self._hidden1(x))
        h2 = F.relu(self._hidden2(h1))
        return self._latnet_mu(h2), self._latent_sigma(h2)


class Decoder(torch.nn.Module):
    """
    Decoder for the variational autoencoder, decodes from the latent space to the output space.
    Forward pass takes a latent space vector and returns an image  vector.
    """

    def __init__(self, latent_size, hidden_size, output_size):
        super(Decoder, self).__init__()

        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden_layer = torch.nn.Linear(latent_size, hidden_size)
        self.output_layer = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.hidden_layer(x))
        return F.relu(self.output_layer(x))


class Vae(torch.nn.Module):

    def __init__(self, encoder, decoder):
        super(Vae, self).__init__()

        # Save dependencies.
        self.encoder = encoder
        self.decoder = decoder

        # Make sure encoder and decoder are compatible.
        assert encoder.latent_size == decoder.latent_size

        # self.sample_from_normal_dist = torch.from_numpy(np.random.normal(0, 1, size=encoder.latent_size.size())).float()

    def _sample_latent(self, mu, log_sigma):
        """
        Return a sample from the latent vector with normal distribution of z ~ N(mu, sigma^2)
        """
        sigma = torch.exp(log_sigma)
        std_z = torch.randn(sigma.size(), device=sigma.device)
        # std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        mu, log_sigma = self.encoder(state)
        z = self._sample_latent(mu, log_sigma)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

    image_dim = 28 * 28
    hidden_dim = 100
    latent_dim = 8
    batch_size = 32
    seed = 1

    # Use GPU?
    use_cuda = torch.cuda.is_available()

    # Seed random number generators.
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    # Use MNIST dataset.
    mnist = torchvision.datasets.MNIST('./', transform=transforms.ToTensor(), download=True)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print(f'Number of samples: {len(mnist)}')

    # Define the network structure.
    encoder = Encoder(image_dim, hidden_dim, hidden_dim, latent_dim).to(device)
    decoder = Decoder(latent_dim, hidden_dim, image_dim).to(device)
    vae = Vae(encoder, decoder).to(device)

    # Use mean L2 norm as loss for the output of the autoencoder.
    generator_loss = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    loss = None
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            input_images, image_classes = data

            image_classes = image_classes.to(device)
            input_images = input_images.resize_(batch_size, image_dim).to(device)

            # Reset gradients to zero
            optimizer.zero_grad()

            # Forward pass
            output_images = vae(input_images)

            # Calculate loss
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = generator_loss(output_images, input_images) + ll

            # Calculate gradients
            loss.backward()

            # Update weights
            optimizer.step()

        print(epoch, loss.data.cpu().numpy())

    plt.imshow(vae(input_images).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
