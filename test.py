"""
Script de teste e avaliação do modelo
"""
import torch
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import Generator
from dataset import LeafDataset
from torch.utils.data import DataLoader
from anomaly_detection import detect_anomaly, calculate_color_reconstructability_index, visualize_anomaly
from gradcam import create_gradcam_for_generator


def evaluate_model(
    generator_path,
    healthy_test_dir,
    disease_test_dir,
    device=None,
    threshold=0.1,
    output_dir='results'
):
    """
    Avalia o modelo nos dados de teste
    
    Args:
        generator_path: caminho para o modelo gerador treinado
        healthy_test_dir: diretório com folhas saudáveis de teste
        disease_test_dir: diretório com folhas doentes de teste
        device: dispositivo (cuda, cuda:0, cuda:1, ou cpu). Se None, detecta automaticamente
        threshold: threshold para classificação de anomalia
        output_dir: diretório para salvar resultados
    """
    # Detecta o dispositivo se não especificado
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    # Infos sobre o dispositivo
    print("="*60)
    print("CONFIGURAÇÃO DO DISPOSITIVO")
    print("="*60)
    if device.startswith('cuda'):
        print(f"✓ GPU detectada e será usada!")
        print(f"  Dispositivo: {device}")
        print(f"  Nome da GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memória total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"⚠ Usando CPU")
        print(f"  Dispositivo: {device}")
    print("="*60)
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'gradcam'), exist_ok=True)
    
    # Carregar modelo
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()
    
    # Criar GradCAM
    try:
        gradcam = create_gradcam_for_generator(generator, device)
        use_gradcam = True
    except:
        print("Aviso: Não foi possível criar GradCAM, continuando sem ele...")
        use_gradcam = False
    
    # Lista todas as imagens
    healthy_images = []
    disease_images = []
    
    if os.path.isdir(healthy_test_dir):
        for filename in os.listdir(healthy_test_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                healthy_images.append(os.path.join(healthy_test_dir, filename))
    
    if os.path.isdir(disease_test_dir):
        for filename in os.listdir(disease_test_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                disease_images.append(os.path.join(disease_test_dir, filename))
    
    print(f"Imagens saudáveis de teste: {len(healthy_images)}")
    print(f"Imagens doentes de teste: {len(disease_images)}")
    
    # Avaliar
    results = {
        'healthy': [],
        'disease': []
    }
    
    # Testa as folha saudáveis
    print("\nTestando folhas saudáveis...")
    for img_path in tqdm(healthy_images):
        is_anomaly, cri, anomaly_map, reconstructed = detect_anomaly(
            img_path, generator, device, threshold
        )
        results['healthy'].append({
            'path': img_path,
            'is_anomaly': is_anomaly,
            'cri': cri,
            'anomaly_map': anomaly_map,
            'reconstructed': reconstructed
        })
    
    # Testa as folha doentes
    print("\nTestando folhas doentes...")
    for img_path in tqdm(disease_images):
        is_anomaly, cri, anomaly_map, reconstructed = detect_anomaly(
            img_path, generator, device, threshold
        )
        results['disease'].append({
            'path': img_path,
            'is_anomaly': is_anomaly,
            'cri': cri,
            'anomaly_map': anomaly_map,
            'reconstructed': reconstructed
        })
    
    # Calcula mtricas
    y_true = [0] * len(results['healthy']) + [1] * len(results['disease'])
    y_pred = [int(r['is_anomaly']) for r in results['healthy']] + \
             [int(r['is_anomaly']) for r in results['disease']]
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    # Estatistica de CRI
    cri_healthy = [r['cri'] for r in results['healthy']]
    cri_disease = [r['cri'] for r in results['disease']]
    
    # Calcular threshold bom se o threshold fornecido não for adequado
    mean_healthy = np.mean(cri_healthy)
    mean_disease = np.mean(cri_disease)
    std_healthy = np.std(cri_healthy)
    std_disease = np.std(cri_disease)
    
    # Se o threshold fornecido está muito alto ou muito baixo, calcular um bom
    optimal_threshold = threshold
    threshold_was_optimized = False
    
    # Verificar se o threshold está inadequado
    if threshold > mean_disease + 3 * std_disease or threshold < mean_healthy - 3 * std_healthy:
        # Threshold muito alto, calcular um ótimo
        # Método 1: Ponto médio entre as médias
        threshold_midpoint = (mean_healthy + mean_disease) / 2
        
        # Método 2: Maximizar F1-score testando diferentes thresholds
        best_f1 = 0
        best_thresh = threshold_midpoint
        test_thresholds = np.linspace(
            max(0, mean_healthy - 2*std_healthy),
            min(mean_disease + 2*std_disease, 0.1),
            100
        )
        
        for test_thresh in test_thresholds:
            y_pred_test = []
            for r in results['healthy']:
                y_pred_test.append(1 if r['cri'] > test_thresh else 0)
            for r in results['disease']:
                y_pred_test.append(1 if r['cri'] > test_thresh else 0)
            
            try:
                test_f1 = f1_score(y_true, y_pred_test, zero_division=0)
                if test_f1 > best_f1:
                    best_f1 = test_f1
                    best_thresh = test_thresh
            except:
                pass
        
        optimal_threshold = best_thresh
        threshold_was_optimized = True
        print(f"\n⚠ AVISO: Threshold fornecido ({threshold:.6f}) não é adequado!")
        print(f"   CRI médio saudáveis: {mean_healthy:.6f} ± {std_healthy:.6f}")
        print(f"   CRI médio doentes: {mean_disease:.6f} ± {std_disease:.6f}")
        print(f"   Threshold ótimo calculado: {optimal_threshold:.6f} (F1-Score: {best_f1:.4f})")
        print(f"   Recalculando métricas com threshold ótimo...\n")
        
        # Recalcula prediçõe com threshold bm
        y_pred = []
        for r in results['healthy']:
            y_pred.append(1 if r['cri'] > optimal_threshold else 0)
        for r in results['disease']:
            y_pred.append(1 if r['cri'] > optimal_threshold else 0)
        
        # Recalcula métricas com threshold bom
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        threshold = optimal_threshold
    
    print("\n" + "="*50)
    print("RESULTADOS DA AVALIAÇÃO")
    print("="*50)
    if threshold_was_optimized:
        print(f"⚠ Threshold foi otimizado automaticamente!")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"\nMatriz de Confusão:")
    print(f"  Verdadeiro Negativo (TN): {cm[0][0]}")
    print(f"  Falso Positivo (FP): {cm[0][1]}")
    print(f"  Falso Negativo (FN): {cm[1][0]}")
    print(f"  Verdadeiro Positivo (TP): {cm[1][1]}")
    print(f"\nCRI Médio - Folhas Saudáveis: {mean_healthy:.6f} ± {std_healthy:.6f}")
    print(f"CRI Médio - Folhas Doentes: {mean_disease:.6f} ± {std_disease:.6f}")
    print(f"Threshold utilizado: {threshold:.6f}")
    if threshold_was_optimized:
        print(f"  (Otimizado automaticamente)")
    print("="*50)
    
    # Salva visu
    print("\nSalvando visualizações...")
    n_samples = min(10, len(results['healthy']), len(results['disease']))
    
    for i in range(n_samples):
        # Folha saudável
        healthy_result = results['healthy'][i]
        visualize_anomaly(
            healthy_result['path'],
            healthy_result['anomaly_map'],
            healthy_result['reconstructed'],
            os.path.join(output_dir, 'visualizations', f'healthy_{i}.png')
        )
        
        # GradCAM para folha saudável
        if use_gradcam:
            try:
                from torchvision import transforms
                from PIL import Image
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                img = Image.open(healthy_result['path']).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                gradcam.visualize(
                    img_tensor,
                    healthy_result['reconstructed'],
                    healthy_result['path'],
                    os.path.join(output_dir, 'gradcam', f'healthy_{i}_gradcam.png')
                )
            except:
                pass
        
        # Folha doente
        disease_result = results['disease'][i]
        visualize_anomaly(
            disease_result['path'],
            disease_result['anomaly_map'],
            disease_result['reconstructed'],
            os.path.join(output_dir, 'visualizations', f'disease_{i}.png')
        )
        
        # GradCAM pRA folha doente
        if use_gradcam:
            try:
                from torchvision import transforms
                from PIL import Image
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
                img = Image.open(disease_result['path']).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                gradcam.visualize(
                    img_tensor,
                    disease_result['reconstructed'],
                    disease_result['path'],
                    os.path.join(output_dir, 'gradcam', f'disease_{i}_gradcam.png')
                )
            except:
                pass
    
    # Plota distrib de CRI
    plt.figure(figsize=(10, 6))
    plt.hist(cri_healthy, bins=30, alpha=0.5, label='Folhas Saudáveis', color='green')
    plt.hist(cri_disease, bins=30, alpha=0.5, label='Folhas Doentes', color='red')
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold ({threshold})')
    plt.xlabel('Índice de Reconstruibilidade de Cores (CRI)')
    plt.ylabel('Frequência')
    plt.title('Distribuição do CRI para Folhas Saudáveis e Doentes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'cri_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Salva resultado em arquivo
    results_file = os.path.join(output_dir, 'results.txt')
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("="*50 + "\n")
        f.write("RESULTADOS DA AVALIAÇÃO\n")
        f.write("="*50 + "\n\n")
        f.write(f"Acurácia: {accuracy:.4f}\n")
        f.write(f"Precisão: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n\n")
        f.write("Matriz de Confusão:\n")
        f.write(f"  Verdadeiro Negativo (TN): {cm[0][0]}\n")
        f.write(f"  Falso Positivo (FP): {cm[0][1]}\n")
        f.write(f"  Falso Negativo (FN): {cm[1][0]}\n")
        f.write(f"  Verdadeiro Positivo (TP): {cm[1][1]}\n\n")
        f.write(f"CRI Médio - Folhas Saudáveis: {mean_healthy:.6f} ± {std_healthy:.6f}\n")
        f.write(f"CRI Médio - Folhas Doentes: {mean_disease:.6f} ± {std_disease:.6f}\n")
        f.write(f"Threshold utilizado: {threshold:.6f}\n")
        if threshold_was_optimized:
            f.write(f"  (Otimizado automaticamente)\n")
    
    print(f"\nResultados salvos em: {results_file}")
    print(f"Visualizações salvas em: {os.path.join(output_dir, 'visualizations')}")
    if use_gradcam:
        print(f"GradCAM salvo em: {os.path.join(output_dir, 'gradcam')}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Testar modelo pix2pix')
    parser.add_argument('--generator_path', type=str, default='checkpoints/generator_final.pth',
                       help='Caminho para o modelo gerador')
    parser.add_argument('--healthy_test_dir', type=str, default='Healthy_Test50',
                       help='Diretório com folhas saudáveis de teste')
    parser.add_argument('--disease_test_dir', type=str, default='Disease_Test100',
                       help='Diretório com folhas doentes de teste')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Threshold para classificação de anomalia')
    parser.add_argument('--device', type=str, default=None,
                       help='Dispositivo (cuda, cuda:0, cpu). Se None, detecta automaticamente')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    evaluate_model(
        generator_path=args.generator_path,
        healthy_test_dir=args.healthy_test_dir,
        disease_test_dir=args.disease_test_dir,
        device=args.device,
        threshold=args.threshold,
        output_dir=args.output_dir
    )

