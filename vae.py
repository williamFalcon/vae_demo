import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torchvision import datasets, transforms
from PIL import Image
import torchvision


class VAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def forward(self, z):
        return self.decode(z)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        loss = self.loss_function(x_hat, x, mu, logvar)

        log = {'train_loss': loss}
        return {'loss':loss, 'log': log}

    def save_image(self, data, filename):
        img = data.clone().clamp(0, 255).numpy()
        img = img[0].transpose(1, 2, 0)
        img = Image.fromarray(img, mode='RGB')
        img.save(filename)

    def validation_step(self, batch, batch_idx):
        x, _ = batch

        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_hat = self(z)
        val_loss = self.loss_function(x_hat, x, mu, logvar)

        return {'val_loss':val_loss, 'x_hat': x_hat}

    def validation_epoch_end(
            self,
            outputs):

        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        x_hat = outputs[-1]['x_hat']

        grid = torchvision.utils.make_grid(x_hat)
        self.logger.experiment.add_image('images', grid, 0)

        log = {'avg_val_loss': val_loss}
        return {'log': log, 'val_loss': log}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.ToTensor()),
            batch_size=self.hparams.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
            batch_size=self.hparams.batch_size)
        return val_loader


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--batch_size', default=32)
    parser.add_argument('--learning_rate', default=1e-3, type=float)

    args = parser.parse_args()

    vae = VAE(hparams=args)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(vae)