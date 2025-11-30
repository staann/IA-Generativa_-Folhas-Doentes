"""
Implementação de Grad-CAM para visualização de explicações
Baseado em: Grad-CAM: Visual Explanations from Deep Networks
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


class GradCAM:
    """Classe para gerar visualizações Grad-CAM"""
    
    def __init__(self, model, target_layer):
        """
        Args:
            model: modelo neural (gerador)
            target_layer: camada alvo para calcular Grad-CAM
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Registrar hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Salva ativações da camada"""
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """Salva gradientes da camada"""
        self.gradients = grad_output[0]
    
    def generate_cam(self, input_image, reconstructed_image):
        """
        Gera o mapa de ativação Grad-CAM
        
        Args:
            input_image: imagem de entrada (tensor)
            reconstructed_image: imagem reconstruída (tensor)
        
        Returns:
            cam: mapa de ativação
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        # Calcular perda (diferença entre entrada e saída)
        loss = F.mse_loss(output, input_image)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Obter gradientes e ativações
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Calcular pesos dos gradientes (média global)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Gerar CAM
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Aplicar ReLU
        cam = np.maximum(cam, 0)
        
        # Normalizar
        cam = cam / (cam.max() + 1e-8)
        
        # Redimensionar para tamanho da imagem original
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        
        return cam
    
    def visualize(self, input_image, reconstructed_image, original_path, output_path):
        """
        Visualiza Grad-CAM sobreposto na imagem original
        
        Args:
            input_image: imagem de entrada (tensor)
            reconstructed_image: imagem reconstruída (tensor)
            original_path: caminho da imagem original
            output_path: caminho para salvar visualização
        """
        # Gerar CAM
        cam = self.generate_cam(input_image, reconstructed_image)
        
        # Carregar imagem original
        original = Image.open(original_path).convert('RGB')
        original = original.resize((256, 256))
        original = np.array(original)
        
        # Aplicar colormap ao CAM
        cam_colored = plt.cm.jet(cam)[:, :, :3]
        cam_colored = (cam_colored * 255).astype(np.uint8)
        
        # Sobrepor CAM na imagem original
        overlay = cv2.addWeighted(original, 0.6, cam_colored, 0.4, 0)
        
        # Criar visualização
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Imagem Original')
        axes[0].axis('off')
        
        axes[1].imshow(cam, cmap='jet')
        axes[1].set_title('Grad-CAM')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Grad-CAM Sobreposto')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()


def create_gradcam_for_generator(generator, device):
    """
    Cria instância de GradCAM para o gerador
    
    Args:
        generator: modelo gerador
        device: dispositivo (cuda ou cpu)
    
    Returns:
        gradcam: instância de GradCAM
    """
    # Usar a última camada convolucional antes da saída
    # No caso do U-Net, podemos usar uma das camadas do decoder
    target_layer = generator.up7.model[0]  # Primeira camada do último upsampling
    
    return GradCAM(generator, target_layer)

