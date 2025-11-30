"""
Script de treinamento do modelo pix2pix
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from models import Generator, Discriminator
from dataset import LeafDataset


def train_pix2pix(
    train_dir,
    epochs=100,
    batch_size=4,
    lr=0.0002,
    device=None,
    save_dir='checkpoints'
):
    """
    Treina o modelo pix2pix
    
    Args:
        train_dir: diretório com imagens de treinamento
        epochs: número de épocas
        batch_size: tamanho do batch
        lr: taxa de aprendizado
        device: dispositivo (cuda, cuda:0, cuda:1, ou cpu). Se None, detecta automaticamente
        save_dir: diretório para salvar checkpoints
    """
    # Detectar dispositivo automaticamente se não especificado
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    # Informações sobre o dispositivo
    print("="*60)
    print("CONFIGURAÇÃO DO DISPOSITIVO")
    print("="*60)
    if device.startswith('cuda'):
        print(f"✓ GPU detectada e será usada!")
        print(f"  Dispositivo: {device}")
        print(f"  Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memória total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Versão CUDA: {torch.version.cuda}")
        print(f"  PyTorch compilado com CUDA: {torch.cuda.is_available()}")
    else:
        print(f"⚠ Usando CPU (GPU não disponível ou não detectada)")
        print(f"  Dispositivo: {device}")
        if not torch.cuda.is_available():
            print(f"  PyTorch não detectou CUDA. Verifique se:")
            print(f"    - PyTorch foi instalado com suporte CUDA")
            print(f"    - Drivers NVIDIA estão atualizados")
            print(f"    - GPU está disponível")
    print("="*60)
    print()
    
    # Criar diretório para checkpoints
    os.makedirs(save_dir, exist_ok=True)
    
    # Dataset e DataLoader
    # Ajustar num_workers baseado no dispositivo
    num_workers = 4 if device.startswith('cuda') else 2
    dataset = LeafDataset(train_dir, mode='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device.startswith('cuda') else False)
    
    # Modelos
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Losses
    criterion_GAN = nn.MSELoss()
    criterion_pixel = nn.L1Loss()
    
    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # Para pix2pix, entrada e saída são a mesma (autoencoder)
    # Mas o discriminador recebe par (entrada, saída)
    
    print(f"Treinando em {device}...")
    print(f"Número de imagens: {len(dataset)}")
    print(f"Épocas: {epochs}, Batch size: {batch_size}")
    
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        epoch_loss_G = 0
        epoch_loss_D = 0
        
        pbar = tqdm(dataloader, desc=f"Época {epoch+1}/{epochs}")
        
        for batch in pbar:
            real_images = batch['image'].to(device)
            
            # ============ Treinar Discriminador ============
            optimizer_D.zero_grad()
            
            # Imagens reais (par de imagens idênticas para folhas saudáveis)
            real_pairs = torch.cat([real_images, real_images], 1)
            pred_real = discriminator(real_images, real_images)
            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            
            # Imagens falsas (geradas)
            fake_images = generator(real_images)
            fake_pairs = torch.cat([real_images, fake_images.detach()], 1)
            pred_fake = discriminator(real_images, fake_images.detach())
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            
            loss_D = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()
            
            # ============ Treinar Gerador ============
            optimizer_G.zero_grad()
            
            # Loss adversarial
            fake_images = generator(real_images)
            pred_fake = discriminator(real_images, fake_images)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            
            # Loss de pixel (L1)
            loss_pixel = criterion_pixel(fake_images, real_images) * 100
            
            # Loss total
            loss_G = loss_GAN + loss_pixel
            loss_G.backward()
            optimizer_G.step()
            
            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()
            
            pbar.set_postfix({
                'Loss_G': f'{loss_G.item():.4f}',
                'Loss_D': f'{loss_D.item():.4f}'
            })
        
        avg_loss_G = epoch_loss_G / len(dataloader)
        avg_loss_D = epoch_loss_D / len(dataloader)
        
        print(f"Época {epoch+1}/{epochs} - Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}")
        
        # Salvar checkpoint a cada 10 épocas
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"Checkpoint salvo: checkpoint_epoch_{epoch+1}.pth")
    
    # Salvar modelo final
    torch.save(generator.state_dict(), os.path.join(save_dir, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, 'discriminator_final.pth'))
    print("Treinamento concluído!")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Treinar modelo pix2pix')
    parser.add_argument('--train_dir', type=str, default='Healthy_Train50',
                       help='Diretório com imagens de treinamento')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamanho do batch')
    parser.add_argument('--lr', type=float, default=0.0002,
                       help='Taxa de aprendizado')
    parser.add_argument('--device', type=str, default=None,
                       help='Dispositivo (cuda, cuda:0, cpu). Se None, detecta automaticamente')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                       help='Diretório para salvar checkpoints')
    
    args = parser.parse_args()
    
    train_pix2pix(
        train_dir=args.train_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )

