"""
Dataset para carregar imagens de folhas
"""
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LeafDataset(Dataset):
    """Dataset para imagens de folhas"""
    
    def __init__(self, root_dir, image_size=256, mode='train'):
        self.root_dir = root_dir
        self.image_size = image_size
        self.mode = mode
        
        # Lista todas as imagens
        self.image_paths = []
        if os.path.isdir(root_dir):
            for filename in os.listdir(root_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(root_dir, filename))
        
        # Transformar
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
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Pra pix2pix- entrada e saída são a mesma imagem (autoencoder)
        image_tensor = self.transform(image)
        
        return {
            'image': image_tensor,
            'path': image_path
        }

