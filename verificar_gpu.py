"""
Script para verificar se a GPU está disponível e configurada corretamente
"""
import torch
import sys

print("="*60)
print("VERIFICAÇÃO DE GPU E CUDA")
print("="*60)

# Verificar PyTorch
print(f"\nVersão do PyTorch: {torch.__version__}")

# Verificar CUDA
print(f"\nCUDA disponível: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ GPU detectada!")
    print(f"\nInformações da GPU:")
    print(f"  Nome: {torch.cuda.get_device_name(0)}")
    print(f"  Versão CUDA: {torch.version.cuda}")
    print(f"  Versão cuDNN: {torch.backends.cudnn.version()}")
    print(f"  Número de GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}:")
        print(f"    Nome: {props.name}")
        print(f"    Memória total: {props.total_memory / 1024**3:.2f} GB")
        print(f"    Memória alocada: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"    Memória reservada: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        print(f"    Compute Capability: {props.major}.{props.minor}")
    
    # Teste rápido
    print(f"\nTestando operação na GPU...")
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ Teste bem-sucedido! GPU está funcionando.")
    except Exception as e:
        print(f"✗ Erro ao testar GPU: {e}")
else:
    print(f"\n✗ CUDA não está disponível!")
    print(f"\nPossíveis causas:")
    print(f"  1. PyTorch foi instalado sem suporte CUDA")
    print(f"  2. Drivers NVIDIA não estão instalados ou desatualizados")
    print(f"  3. GPU não é compatível com CUDA")
    print(f"\nPara instalar PyTorch com CUDA, visite:")
    print(f"  https://pytorch.org/get-started/locally/")
    print(f"\nPara RTX 4080, você precisa:")
    print(f"  - PyTorch com CUDA 11.8 ou 12.1")
    print(f"  - Drivers NVIDIA atualizados (versão 525 ou superior)")

print("\n" + "="*60)

