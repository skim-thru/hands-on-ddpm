import os
import torch
import configargparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from noise_scheduler import NoiseScheduler
from tqdm import tqdm
import numpy as np
from model import UNetModel
import torchvision
from torchvision.utils import make_grid

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_config():
    parser = configargparse.ArgumentParser(
        description="Training Configuration",
        default_config_files=['config.yaml']
    )

    parser.add('--config', is_config_file=True, help='config file path')
    parser.add('--num_epochs', type=int, default=2046, help='Number of epochs to train')
    parser.add('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add('--learning_rate', type=float, default=2e-4, help='Learning rate for optimizer')
    parser.add('--beta_1', type=float, default=1e-4, help='Beta 1 for noise scheduler')
    parser.add('--beta_T', type=float, default=0.02, help='Beta T for noise scheduler')
    parser.add('--save_epoch', type=int, default=100, help='Save model every n epochs')
    parser.add('--resume_from', type=str, help='Path to resume from checkpoint')
    parser.add('--log_dir', type=str, default='./runs', help='Directory for TensorBoard logs')
    parser.add('--output_dir', type=str, default='./outputs/cifar10', help='Directory for saving checkpoints')

    parser.add('--schedule', type=str, default='linear', help='Noise scheduler schedule type')
    parser.add('--T', type=int, default=1000, help='Number of timesteps for noise scheduler')

    parser.add('--ch', type=int, default=128, help='Number of base channels for UNet model')
    parser.add('--ch_mult', type=int, nargs='+', default=[1, 2, 2, 2], help='Channel multiplier for UNet model')
    parser.add('--num_res_blocks', type=int, default=2, help='Number of residual blocks per stage')
    parser.add('--attn_resolutions', type=int, nargs='+', default=[16], help='Resolutions to apply attention')
    parser.add('--dropout', type=float, default=0.1, help='Dropout rate for UNet model')
    parser.add('--resamp_with_conv', type=bool, default=True, help='Use convolutional resampling in UNet model')
    parser.add('--num_classes', type=int, default=10, help='Number of classes (output channels) for UNet model')

    parser.add('--num_samples', type=int, default=100, help='Number of samples to generate')
    parser.add('--image_size', type=int, default=100, help='Size of samples to generate')
    parser.add('--nrow', type=int, default=100, help='Number of rows when making a grid image')

    return parser.parse_args()

def train(config):
    writer = SummaryWriter(log_dir=config.log_dir)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    os.makedirs(config.output_dir, exist_ok=True)

    noise_scheduler = NoiseScheduler(
        schedule=config.schedule,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        T=config.T,
        device=device
    )

    model = UNetModel(
        ch=config.ch,
        ch_mult=config.ch_mult,
        num_res_blocks=config.num_res_blocks,
        attn_resolutions=config.attn_resolutions,
        dropout=config.dropout,
        resamp_with_conv=config.resamp_with_conv,
        num_classes=config.num_classes,
    ).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = torch.nn.MSELoss()

    start_epoch = 1

    if config.resume_from:
        checkpoint = torch.load(config.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    for epoch in range(start_epoch, config.num_epochs + 1):
        model.train()
        losses = []
        for images, _ in tqdm(train_loader):
            optimizer.zero_grad()
            images = images.to(device)

            noise = torch.randn_like(images).to(device)
            t = torch.randint(0, config.T, (images.shape[0],)).to(device)

            noisy_images = noise_scheduler.step_forward(images, noise, t)
            noise_pred = model(noisy_images, t)
            loss = criterion(noise_pred, noise).mean()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = np.mean(losses)
        print(f"Epoch [{epoch}/{config.num_epochs}] Loss: {avg_loss:.4f}")

        writer.add_scalar('Loss/train', avg_loss, epoch)

        if epoch % config.save_epoch == 0 or epoch == config.num_epochs:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(checkpoint, f'{config.output_dir}/ddpm_cifar10_{epoch:04d}.pth')

            x_0 = sample(model=model, noise_scheduler=noise_scheduler, T=config.T, num_samples=config.num_samples, image_size=config.image_size)
            grid = visualize(x_0, nrow=config.nrow)
            writer.add_image('Generated Images', grid, epoch)

    writer.close()

def sample(model: UNetModel, noise_scheduler: NoiseScheduler, T: int = 1000, num_samples: int = 100, image_size: int = 32):
    model.eval()
    x_t = torch.randn((num_samples, 3, image_size, image_size), device=device)
    for t in tqdm(reversed(range(T)), desc="Sampling"):
        timestep = torch.full((num_samples,), t, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(x_t, timestep)
            x_t = noise_scheduler.step_reverse(images=x_t, noise_pred=noise_pred, t=t)
    return x_t

def visualize(x_t, nrow):
    images = torch.clamp(x_t * 0.5 + 0.5, 0, 1).detach().cpu()
    grid = make_grid(images, nrow=nrow)
    return grid

if __name__ == '__main__':
    config = get_config()
    train(config)
