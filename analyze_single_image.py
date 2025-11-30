
#Script para analisar uma única imagem de folha
#Use: python analyze_single_image.py caminho/da/imagem.jpg

import torch
import sys
import os
from models import Generator
from anomaly_detection import detect_anomaly, visualize_anomaly


def analyze_leaf(image_path, generator_path='checkpoints/generator_final.pth', threshold=0.005178):
    """
    Analisa uma única imagem de folha e mostra o resultado
    
    Args:
        image_path: caminho para a imagem da folha
        generator_path: caminho para o modelo treinado
        threshold: threshold para classificação (0.005178 é o otimizado)
    """
    
    # Detectar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Carregar modelo
    print("\nCarregando modelo")
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    
    # Detectar anomalia
    print("Analisando imagem")
    is_anomaly, cri, anomaly_map, reconstructed = detect_anomaly(
        image_path, generator, device, threshold
    )
    
    # Mostrar resultados
    print("\n" + "="*60)
    print("RESULTADO DA ANÁLISE")
    print("="*60)
    print(f"Índice de Reconstruibilidade (CRI): {cri:.6f}")
    print(f"Threshold utilizado: {threshold:.6f}")
    print()
    
    if is_anomaly:
        print(" DIAGNÓSTICO: FOLHA DOENTE (Anomalia Detectada)")
        print(f"   O CRI ({cri:.6f}) está ACIMA do threshold ({threshold:.6f})")
        print(f"   Diferença: +{((cri/threshold - 1) * 100):.1f}%")
        print()
    else:
        print(" DIAGNÓSTICO: FOLHA SAUDÁVEL (Sem Anomalias)")
        print(f"   O CRI ({cri:.6f}) está ABAIXO do threshold ({threshold:.6f})")
        print(f"   Diferença: {((cri/threshold - 1) * 100):.1f}%")
        print()
    
    print("="*60)
    
    # Salvar visualização
    output_path = image_path.rsplit('.', 1)[0] + '_analysis.png'
    visualize_anomaly(image_path, anomaly_map, reconstructed, output_path)
    print(f"\n Visualização salva em: {output_path}")
    
    return is_anomaly, cri


if __name__ == '__main__':  
    image_path = sys.argv[1]
    
    # Verificar se threshold foi fornecido
    threshold = 0.005178  # Threshold otimizado
    if len(sys.argv) > 3 and sys.argv[2] == '--threshold':
        try:
            threshold = float(sys.argv[3])
            print(f"")
        except:
            print("Threshold inválido")
    
    analyze_leaf(image_path, threshold=threshold)
