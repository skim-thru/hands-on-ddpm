import torch

class NoiseScheduler:
    def __init__(self, schedule: str = "linear", beta_1: float = 1e-4, beta_T: float = 0.02, T: int = 1000, device="cuda:0"):
        if schedule == "linear":
            self.betas = torch.linspace(start=beta_1, end=beta_T, steps=T).to(device=device)
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0).to(device=device)

    # Algorithm 1
    def step_forward(self, images, noise, t):
        return torch.sqrt(self.alphas_bar[t].view(-1, 1, 1, 1)) * images + torch.sqrt(1. - self.alphas_bar[t].view(-1, 1, 1, 1)) * noise

    # Algorithm 2
    def step_reverse(self, images, noise_pred, t):
        # images: x_t
        # noise_pred: \epsilon_{\theta}(\mathbf{x}_{t}, t)
        mean = (1. - self.alphas[t].view(-1, 1, 1, 1)) / torch.sqrt(1. - self.alphas_bar[t].view(-1, 1, 1, 1))
        mean = mean * noise_pred
        mean = images - mean
        mean = (1. / torch.sqrt(self.alphas[t].view(-1, 1, 1, 1))) * mean

        if t == 0:
            return mean

        z = torch.randn_like(images)
        variance = (1. - self.alphas_bar[t - 1].view(-1, 1, 1, 1)) / (1. - self.alphas_bar[t].view(-1, 1, 1, 1))
        variance = variance * self.betas[t].view(-1, 1, 1, 1)
        sigma = torch.sqrt(variance)

        return mean + sigma * z


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import datasets, transforms
    import matplotlib.pyplot as plt

    noise_scheduler = NoiseScheduler()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    def denormalize(tensor, mean, std):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return torch.clamp(tensor, 0, 1)

    def show_images(images, title):
        images = denormalize(images, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        grid_img = torchvision.utils.make_grid(images.cpu(), nrow=8)
        plt.figure(figsize=(10, 10))
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.title(title)
        plt.axis('off')
        plt.savefig(f"{title}.png")

    data_iter = iter(train_loader)
    images, _ = next(data_iter)
    images = images.to("cuda:0")

    timesteps = [0, 250, 500, 750, 999]
    noisy_images_list = []

    for t in timesteps:
        noisy_images = noise_scheduler.step_forward(images, t)
        noisy_images_list.append((t, noisy_images))

    show_images(images, 'Original Images')

    for t, noisy_images in noisy_images_list:
        show_images(noisy_images, f'Noisy Images at Step {t}')
