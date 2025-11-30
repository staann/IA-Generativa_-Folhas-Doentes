"""
Detecção de anomalias baseada em reconstruibilidade de cores
Baseado no artigo: Image-based plant disease diagnosis with unsupervised anomaly detection
"""
import torch
import numpy as np
from PIL import Image
import cv2


def calculate_color_reconstructability_index(original, reconstructed):
    """
    Calcula o índice de reconstruibilidade de cores (CRI)
    Quanto maior o CRI, maior a anomalia detectada
    
    Args:
        original: imagem original (numpy array ou tensor)
        reconstructed: imagem reconstruída pelo gerador (numpy array ou tensor)
    
    Returns:
        cri: índice de reconstruibilidade de cores
        anomaly_map: mapa de anomalias
    """
    # Converter para numpy se necessário
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.cpu().numpy()
    
    # Garantir que está no formato correto (H, W, C)
    if len(original.shape) == 4:
        original = original[0].transpose(1, 2, 0)
    if len(reconstructed.shape) == 4:
        reconstructed = reconstructed[0].transpose(1, 2, 0)
    
    # Normalizar para [0, 1] se necessário
    if original.max() > 1.0:
        original = original / 255.0
    if reconstructed.max() > 1.0:
        reconstructed = reconstructed / 255.0
    
    # Converter de [-1, 1] para [0, 1] se necessário (saída do tanh)
    if original.min() < 0:
        original = (original + 1) / 2
    if reconstructed.min() < 0:
        reconstructed = (reconstructed + 1) / 2
    
    # Calcular diferença em cada canal de cor
    diff_r = np.abs(original[:, :, 0] - reconstructed[:, :, 0])
    diff_g = np.abs(original[:, :, 1] - reconstructed[:, :, 1])
    diff_b = np.abs(original[:, :, 2] - reconstructed[:, :, 2])
    
    # Média das diferenças (anomaly map)
    anomaly_map = (diff_r + diff_g + diff_b) / 3.0
    
    # CRI: média do mapa de anomalias
    cri = np.mean(anomaly_map)
    
    return cri, anomaly_map


def detect_anomaly(image_path, generator, device, threshold=0.1):
    """
    Detecta anomalias em uma imagem de folha
    
    Args:
        image_path: caminho para a imagem
        generator: modelo gerador treinado
        device: dispositivo (cuda ou cpu)
        threshold: threshold para classificação de anomalia
    
    Returns:
        is_anomaly: True se detectou anomalia
        cri: índice de reconstruibilidade de cores
        anomaly_map: mapa de anomalias
        reconstructed: imagem reconstruída
    """
    # Carregar e pré-processar imagem
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256, 256))
    
    # Converter para tensor
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Gerar reconstrução
    generator.eval()
    with torch.no_grad():
        reconstructed = generator(img_tensor)
    
    # Calcular CRI
    cri, anomaly_map = calculate_color_reconstructability_index(
        img_tensor, reconstructed
    )
    
    # Classificar como anomalia se CRI > threshold
    is_anomaly = cri > threshold
    
    return is_anomaly, cri, anomaly_map, reconstructed


def visualize_anomaly(original_path, anomaly_map, reconstructed, output_path):
    """
    Visualiza o mapa de anomalias sobreposto na imagem original
    
    Args:
        original_path: caminho da imagem original
        anomaly_map: mapa de anomalias (numpy array)
        reconstructed: imagem reconstruída (tensor)
        output_path: caminho para salvar visualização
    """
    import matplotlib.pyplot as plt
    
    # Carregar imagem original
    original = Image.open(original_path).convert('RGB')
    original = original.resize((256, 256))
    original = np.array(original)
    
    # Converter reconstructed para numpy
    if isinstance(reconstructed, torch.Tensor):
        recon_np = reconstructed[0].cpu().numpy().transpose(1, 2, 0)
        recon_np = (recon_np + 1) / 2  # De [-1, 1] para [0, 1]
        recon_np = (recon_np * 255).astype(np.uint8)
    else:
        recon_np = reconstructed
    
    # Normalizar anomaly_map para visualização
    anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    anomaly_map_colored = plt.cm.jet(anomaly_map_norm)[:, :, :3]
    anomaly_map_colored = (anomaly_map_colored * 255).astype(np.uint8)
    
    # Criar visualização
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(recon_np)
    axes[1].set_title('Reconstruída')
    axes[1].axis('off')
    
    axes[2].imshow(anomaly_map, cmap='hot')
    axes[2].set_title('Mapa de Anomalias')
    axes[2].axis('off')
    
    # Sobrepor mapa de anomalias na imagem original
    overlay = cv2.addWeighted(original, 0.6, anomaly_map_colored, 0.4, 0)
    axes[3].imshow(overlay)
    axes[3].set_title('Sobreposição')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

