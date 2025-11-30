"""
Script principal para executar o projeto completo
"""
import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Executar projeto completo')
    parser.add_argument('--mode', type=str, choices=['train', 'test', 'both'],
                       default='both', help='Modo de execução')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas para treinamento')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Tamanho do batch')
    parser.add_argument('--threshold', type=float, default=0.1,
                       help='Threshold para classificação de anomalia')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("="*60)
        print("INICIANDO TREINAMENTO")
        print("="*60)
        os.system(f'python train.py --train_dir Healthy_Train50 --epochs {args.epochs} --batch_size {args.batch_size}')
        print("\n")
    
    if args.mode in ['test', 'both']:
        print("="*60)
        print("INICIANDO TESTE E AVALIAÇÃO")
        print("="*60)
        generator_path = 'checkpoints/generator_final.pth'
        if not os.path.exists(generator_path):
            print(f"AVISO: {generator_path} não encontrado!")
            print("Tentando usar último checkpoint...")
            checkpoints = [f for f in os.listdir('checkpoints') if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                # Pegar o checkpoint mais recente
                checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                latest = checkpoints[-1]
                print(f"Usando: {latest}")
                # Extrair o generator do checkpoint
                import torch
                checkpoint = torch.load(os.path.join('checkpoints', latest), map_location='cpu')
                torch.save(checkpoint['generator_state_dict'], generator_path)
                print(f"Generator extraído e salvo em {generator_path}")
            else:
                print("ERRO: Nenhum checkpoint encontrado!")
                sys.exit(1)
        
        os.system(f'python test.py --generator_path {generator_path} --healthy_test_dir Healthy_Test50 --disease_test_dir Disease_Test100 --threshold {args.threshold}')
        print("\n")
        print("="*60)
        print("PROJETO CONCLUÍDO!")
        print("="*60)
        print("Verifique os resultados em:")
        print("  - results/results.txt")
        print("  - results/visualizations/")
        print("  - results/gradcam/")


if __name__ == '__main__':
    main()

