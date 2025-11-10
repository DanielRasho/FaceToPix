import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import itertools
import numpy as np
from tqdm import tqdm


# ==================== GENERADOR ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Encoder
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(
                    in_features, out_features, 3, stride=2, padding=1, output_padding=1
                ),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, output_nc, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ==================== DISCRIMINADOR ====================
class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(input_nc, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)


# ==================== DATASET ====================
class ImageDataset(Dataset):
    def __init__(self, root_pixelated, root_sharp, transform=None):
        self.transform = transform
        self.files_pixelated = sorted(
            [
                os.path.join(root_pixelated, f)
                for f in os.listdir(root_pixelated)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        self.files_sharp = sorted(
            [
                os.path.join(root_sharp, f)
                for f in os.listdir(root_sharp)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    def __getitem__(self, index):
        img_blur = Image.open(self.files_pixelated[index % len(self.files_pixelated)]).convert(
            "RGB"
        )
        img_sharp = Image.open(self.files_sharp[index % len(self.files_sharp)]).convert(
            "RGB"
        )

        if self.transform:
            img_blur = self.transform(img_blur)
            img_sharp = self.transform(img_sharp)

        return {"blur": img_blur, "sharp": img_sharp}

    def __len__(self):
        return max(len(self.files_pixelated), len(self.files_sharp))


# ==================== REPLAY BUFFER ====================
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if np.random.uniform(0, 1) > 0.5:
                    i = np.random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


# ==================== FUNCIONES DE PÉRDIDA ====================
def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


# ==================== ENTRENAMIENTO ====================
class CycleGANTrainer:
    def __init__(
        self,
        blur_path,
        sharp_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = device

        # Inicializar generadores y discriminadores
        self.G_blur2sharp = Generator().to(device)  # Blur -> Sharp
        self.G_sharp2blur = Generator().to(device)  # Sharp -> Blur
        self.D_sharp = Discriminator().to(device)
        self.D_blur = Discriminator().to(device)

        # Inicializar pesos
        self.G_blur2sharp.apply(weights_init_normal)
        self.G_sharp2blur.apply(weights_init_normal)
        self.D_sharp.apply(weights_init_normal)
        self.D_blur.apply(weights_init_normal)

        # Optimizadores
        self.optimizer_G = optim.Adam(
            itertools.chain(
                self.G_blur2sharp.parameters(), self.G_sharp2blur.parameters()
            ),
            lr=0.0002,
            betas=(0.5, 0.999),
        )
        self.optimizer_D_sharp = optim.Adam(
            self.D_sharp.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )
        self.optimizer_D_blur = optim.Adam(
            self.D_blur.parameters(), lr=0.0002, betas=(0.5, 0.999)
        )

        # Pérdidas
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()

        # Buffers
        self.fake_sharp_buffer = ReplayBuffer()
        self.fake_blur_buffer = ReplayBuffer()

        # Dataset
        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        dataset = ImageDataset(blur_path, sharp_path, transform=transform)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    def train_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch}")

        for i, batch in enumerate(progress_bar):
            real_blur = batch["blur"].to(self.device)
            real_sharp = batch["sharp"].to(self.device)

            # Target labels
            valid = torch.ones((real_blur.size(0), 1, 8, 8), requires_grad=False).to(
                self.device
            )
            fake = torch.zeros((real_blur.size(0), 1, 8, 8), requires_grad=False).to(
                self.device
            )

            # ==================== Entrenar Generadores ====================
            self.optimizer_G.zero_grad()

            # Identity loss
            loss_id_sharp = self.criterion_identity(
                self.G_blur2sharp(real_sharp), real_sharp
            )
            loss_id_blur = self.criterion_identity(
                self.G_sharp2blur(real_blur), real_blur
            )

            # GAN loss
            fake_sharp = self.G_blur2sharp(real_blur)
            loss_GAN_blur2sharp = self.criterion_GAN(self.D_sharp(fake_sharp), valid)

            fake_blur = self.G_sharp2blur(real_sharp)
            loss_GAN_sharp2blur = self.criterion_GAN(self.D_blur(fake_blur), valid)

            # Cycle loss
            recovered_blur = self.G_sharp2blur(fake_sharp)
            loss_cycle_blur = self.criterion_cycle(recovered_blur, real_blur)

            recovered_sharp = self.G_blur2sharp(fake_blur)
            loss_cycle_sharp = self.criterion_cycle(recovered_sharp, real_sharp)

            # Total loss
            loss_G = (
                loss_GAN_blur2sharp
                + loss_GAN_sharp2blur
                + 10.0 * (loss_cycle_blur + loss_cycle_sharp)
                + 5.0 * (loss_id_sharp + loss_id_blur)
            )

            loss_G.backward()
            self.optimizer_G.step()

            # ==================== Entrenar Discriminador Sharp ====================
            self.optimizer_D_sharp.zero_grad()

            loss_real = self.criterion_GAN(self.D_sharp(real_sharp), valid)
            fake_sharp = self.fake_sharp_buffer.push_and_pop(fake_sharp)
            loss_fake = self.criterion_GAN(self.D_sharp(fake_sharp.detach()), fake)

            loss_D_sharp = (loss_real + loss_fake) / 2
            loss_D_sharp.backward()
            self.optimizer_D_sharp.step()

            # ==================== Entrenar Discriminador Blur ====================
            self.optimizer_D_blur.zero_grad()

            loss_real = self.criterion_GAN(self.D_blur(real_blur), valid)
            fake_blur = self.fake_blur_buffer.push_and_pop(fake_blur)
            loss_fake = self.criterion_GAN(self.D_blur(fake_blur.detach()), fake)

            loss_D_blur = (loss_real + loss_fake) / 2
            loss_D_blur.backward()
            self.optimizer_D_blur.step()

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "G": f"{loss_G.item():.4f}",
                    "D_S": f"{loss_D_sharp.item():.4f}",
                    "D_B": f"{loss_D_blur.item():.4f}",
                }
            )

    def train(self, epochs=200):
        for epoch in range(epochs):
            self.train_epoch(epoch)

            # Guardar modelos cada 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_models(f"checkpoint_epoch_{epoch+1}.pth")

    def save_models(self, path):
        torch.save(
            {
                "G_blur2sharp": self.G_blur2sharp.state_dict(),
                "G_sharp2blur": self.G_sharp2blur.state_dict(),
                "D_sharp": self.D_sharp.state_dict(),
                "D_blur": self.D_blur.state_dict(),
            },
            path,
        )
        print(f"Modelos guardados en {path}")

    def load_models(self, path):
        checkpoint = torch.load(path)
        self.G_blur2sharp.load_state_dict(checkpoint["G_blur2sharp"])
        self.G_sharp2blur.load_state_dict(checkpoint["G_sharp2blur"])
        self.D_sharp.load_state_dict(checkpoint["D_sharp"])
        self.D_blur.load_state_dict(checkpoint["D_blur"])
        print(f"Modelos cargados desde {path}")

    def transform_image(self, image_path, output_path, sharp_to_blur=True):
        """Transforma una imagen nítida a borrosa (o viceversa)"""
        if sharp_to_blur:
            generator = self.G_sharp2blur
            generator.eval()
        else:
            generator = self.G_blur2sharp
            generator.eval()

        transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        img = Image.open(image_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = generator(img_tensor)

        # Denormalizar y guardar
        output = output.squeeze(0).cpu()
        output = output * 0.5 + 0.5
        output = transforms.ToPILImage()(output)
        output.save(output_path)
        direction = "nítida → borrosa" if sharp_to_blur else "borrosa → nítida"
        print(f"Imagen transformada ({direction}) guardada en {output_path}")


# ==================== USO ====================
if __name__ == "__main__":
    # Configurar rutas
    PIXEL_PATH = "path/to/blur/images"
    SHARP_PATH = "path/to/sharp/images"

    # Crear trainer
    trainer = CycleGANTrainer(PIXEL_PATH, SHARP_PATH)

    # Entrenar
    trainer.train(epochs=200)

    # Transformar imágenes de prueba
    # trainer.load_models('checkpoint_epoch_200.pth')

    # De nítida a borrosa
    # trainer.transform_image('test_sharp.jpg', 'output_blur.jpg', sharp_to_blur=True)

    # De borrosa a nítida
    # trainer.transform_image('test_blur.jpg', 'output_sharp.jpg', sharp_to_blur=False)
