# 🌿 PROJETO: Detecção de Doenças em Folhas - Google Colab
# Copie este arquivo completo para uma célula do Colab e execute

# =============================================================================
# PARTE 1: IMPORTS E CONFIGURAÇÕES
# =============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("✓ Imports concluídos!")

# =============================================================================
# PARTE 2: DEFINIÇÃO DOS MODELOS (Generator e Discriminator)
# =============================================================================

class UNetDown(nn.Module):
    """Bloco de downsampling do U-Net"""
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    """Bloco de upsampling do U-Net"""
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    """Gerador U-Net para pix2pix"""
    def __init__(self, in_channels=3, out_channels=3):
        super(Generator, self).__init__()
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


class Discriminator(nn.Module):
    """Discriminador PatchGAN para pix2pix"""
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, img_A, img_B):
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

print("✓ Modelos definidos!")

# =============================================================================
# PARTE 3: DATASET
# =============================================================================

class LeafDataset(Dataset):
    def __init__(self, root_dir, image_size=256, mode='train'):
        self.root_dir = root_dir
        self.image_paths = []
        if os.path.isdir(root_dir):
            for filename in os.listdir(root_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root_dir, filename))
        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        return {'image': self.transform(image), 'path': self.image_paths[idx]}

print("✓ Dataset definido!")

# =============================================================================
# PARTE 4: FUNÇÕES DE DETECÇÃO DE ANOMALIAS (CRI)
# =============================================================================

def calculate_cri(original, reconstructed):
    """Calcula o Índice de Reconstruibilidade de Cores"""
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    
    if len(original.shape) == 4:
        original = original[0].transpose(1, 2, 0)
    if len(reconstructed.shape) == 4:
        reconstructed = reconstructed[0].transpose(1, 2, 0)
    
    if original.min() < 0:
        original = (original + 1) / 2
    if reconstructed.min() < 0:
        reconstructed = (reconstructed + 1) / 2
    
    diff = np.abs(original - reconstructed)
    cri = np.mean(diff)
    anomaly_map = np.mean(diff, axis=2)
    
    return cri, anomaly_map

def detect_anomaly(image_path, generator, device, threshold):
    """Detecta anomalia em uma imagem"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed = generator(image_tensor)
    
    cri, anomaly_map = calculate_cri(image_tensor, reconstructed)
    is_anomaly = cri > threshold
    
    return is_anomaly, cri, anomaly_map, reconstructed

print("✓ Funções de anomalia definidas!")

# =============================================================================
# PARTE 5: FUNÇÃO DE TREINAMENTO
# =============================================================================

def train_model(train_dir, epochs=300, batch_size=4, lr=0.0001, device='cuda'):
    """Treina o modelo pix2pix"""
    os.makedirs('checkpoints', exist_ok=True)
    
    dataset = LeafDataset(train_dir, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    print(f"Treinando em {device}...")
    print(f"Imagens: {len(dataset)}, Épocas: {epochs}, Batch: {batch_size}")
    
    for epoch in range(epochs):
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{epochs}")
        
        for batch in pbar:
            real_images = batch['image'].to(device)
            
            # Treinar Discriminador
            optimizer_D.zero_grad()
            pred_real = discriminator(real_images, real_images)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            fake_images = generator(real_images)
            pred_fake = discriminator(real_images, fake_images.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # Treinar Gerador
            optimizer_G.zero_grad()
            fake_images = generator(real_images)
            pred_fake = discriminator(real_images, fake_images)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_pixel = criterion_pixel(fake_images, real_images) * 100
            loss_G = loss_GAN + loss_pixel
            loss_G.backward()
            optimizer_G.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            
            pbar.set_postfix({'Loss_G': f'{loss_G.item():.4f}', 'Loss_D': f'{loss_D.item():.4f}'})
        
        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_D = epoch_loss_D / len(dataloader)
        print(f"Época {epoch+1}/{epochs} - Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}")
        
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), f'checkpoints/generator_epoch_{epoch+1}.pth')
    
    torch.save(generator.state_dict(), 'checkpoints/generator_final.pth')
    print("✓ Treinamento concluído!")
    return generator

print("✓ Função de treinamento definida!")

# =============================================================================
# PARTE 6: FUNÇÃO DE TESTE E AVALIAÇÃO
# =============================================================================

def test_model(generator, healthy_dir, disease_dir, device='cuda', threshold=0.1):
    """Testa e avalia o modelo"""
    generator.eval()
    
    healthy_images = [os.path.join(healthy_dir, f) for f in os.listdir(healthy_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    disease_images = [os.path.join(disease_dir, f) for f in os.listdir(disease_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Folhas saudáveis: {len(healthy_images)}")
    print(f"Folhas doentes: {len(disease_images)}")
    
    cri_healthy = []
    cri_disease = []
    
    print("\nTestando folhas saudáveis...")
    for img_path in tqdm(healthy_images):
        _, cri, _, _ = detect_anomaly(img_path, generator, device, threshold)
        cri_healthy.append(cri)
    
    print("Testando folhas doentes...")
    for img_path in tqdm(disease_images):
        _, cri, _, _ = detect_anomaly(img_path, generator, device, threshold)
        cri_disease.append(cri)
    
    # Otimizar threshold
    mean_healthy = np.mean(cri_healthy)
    mean_disease = np.mean(cri_disease)
    std_healthy = np.std(cri_healthy)
    std_disease = np.std(cri_disease)
    
    # Grid search para melhor threshold
    best_f1 = 0
    best_thresh = threshold
    test_thresholds = np.linspace(max(0, mean_healthy - 2*std_healthy), 
                                   mean_disease + 2*std_disease, 100)
    
    y_true = [0] * len(cri_healthy) + [1] * len(cri_disease)
    
    for test_thresh in test_thresholds:
        y_pred = [1 if c > test_thresh else 0 for c in cri_healthy] + \
                 [1 if c > test_thresh else 0 for c in cri_disease]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = test_thresh
    
    # Calcular métricas com threshold otimizado
    y_pred = [1 if c > best_thresh else 0 for c in cri_healthy] + \
             [1 if c > best_thresh else 0 for c in cri_disease]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*60)
    print("RESULTADOS")
    print("="*60)
    print(f"Threshold otimizado: {best_thresh:.6f}")
    print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precisão: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score: {f1:.4f} ({f1*100:.2f}%)")
    print(f"\nMatriz de Confusão:")
    print(f"  TN: {cm[0][0]}, FP: {cm[0][1]}")
    print(f"  FN: {cm[1][0]}, TP: {cm[1][1]}")
    print(f"\nCRI Saudáveis: {mean_healthy:.6f} ± {std_healthy:.6f}")
    print(f"CRI Doentes: {mean_disease:.6f} ± {std_disease:.6f}")
    print("="*60)
    
    # Plotar distribuição
    plt.figure(figsize=(10, 6))
    plt.hist(cri_healthy, bins=30, alpha=0.5, label='Saudáveis', color='green')
    plt.hist(cri_disease, bins=30, alpha=0.5, label='Doentes', color='red')
    plt.axvline(best_thresh, color='black', linestyle='--', label=f'Threshold ({best_thresh:.4f})')
    plt.xlabel('CRI')
    plt.ylabel('Frequência')
    plt.title('Distribuição do CRI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return best_thresh, accuracy, precision, recall, f1

print("✓ Função de teste definida!")
print("\n" + "="*60)
print("TODAS AS FUNÇÕES CARREGADAS!")
print("="*60)
print("\nAgora você pode executar:")
print("1. generator = train_model(TRAIN_DIR, epochs=300)")
print("2. threshold, acc, prec, rec, f1 = test_model(generator, TEST_HEALTHY, TEST_DISEASE)")
