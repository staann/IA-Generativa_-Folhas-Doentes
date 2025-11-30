# Guia para Usar GPU NVIDIA RTX 4060

## 1. Verificar se a GPU está disponível

Primeiro, execute o script de verificação:

```bash
python verificar_gpu.py
```

Este script mostrará:
- Se o PyTorch detecta sua GPU
- Informações sobre a GPU (nome, memória, etc.)
- Se há algum problema de configuração

## 2. Instalar PyTorch com Suporte CUDA

Para usar sua RTX 4080, você precisa instalar o PyTorch com suporte CUDA.

### Opção A: Instalação via pip (Recomendado)

Para RTX 4080, você pode usar CUDA 11.8 ou 12.1:

**CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Opção B: Verificar versão CUDA instalada

Primeiro, verifique qual versão do CUDA você tem instalada:

```bash
nvidia-smi
```

Procure pela linha "CUDA Version" no output. Depois, instale o PyTorch compatível visitando:
https://pytorch.org/get-started/locally/

## 3. Verificar Instalação

Após instalar, execute novamente:

```bash
python verificar_gpu.py
```

Você deve ver algo como:
```
✓ GPU detectada!
  Nome: NVIDIA GeForce RTX 4060
  Versão CUDA: 11.8 (ou 12.1)
```

## 4. Usar GPU no Treinamento

O código **já está configurado para usar GPU automaticamente**! Basta executar:

```bash
python train.py --train_dir Healthy_Train50 --epochs 100 --batch_size 4
```

O script detectará automaticamente sua GPU e mostrará:
```
============================================================
CONFIGURAÇÃO DO DISPOSITIVO
============================================================
✓ GPU detectada e será usada!
  Dispositivo: cuda
  Nome da GPU: NVIDIA GeForce RTX 4060
  Memória total: 8.00 GB
  Versão CUDA: 11.8
  PyTorch compilado com CUDA: True
============================================================
```

## 5. Forçar Uso de GPU Específica (Opcional)

Se você tiver múltiplas GPUs, pode especificar qual usar:

```bash
python train.py --device cuda:0 --train_dir Healthy_Train50 --epochs 100
```

Ou para usar a segunda GPU:
```bash
python train.py --device cuda:1 --train_dir Healthy_Train50 --epochs 100
```

## 6. Otimizações para RTX 4060

Com uma RTX 4060 (8GB de VRAM), você pode:

- **Aumentar o batch_size** para acelerar o treinamento:
  ```bash
  python train.py --batch_size 4 --train_dir Healthy_Train50 --epochs 100
  ```
  Com 8GB de VRAM, você pode tentar aumentar gradualmente:
  ```bash
  python train.py --batch_size 6 --train_dir Healthy_Train50 --epochs 100
  ```
  Ou até:
  ```bash
  python train.py --batch_size 8 --train_dir Healthy_Train50 --epochs 100
  ```
  
  **Nota:** Se receber erro de "Out of memory", reduza o batch_size para 2 ou 4.

- **Usar mixed precision** (opcional, requer modificação do código)

## 7. Verificar Uso da GPU Durante Treinamento

Para monitorar o uso da GPU durante o treinamento, abra outro terminal e execute:

```bash
nvidia-smi -l 1
```

Isso mostrará o uso da GPU atualizado a cada segundo.

## 8. Troubleshooting

### Problema: "CUDA not available"

**Solução 1:** Verifique se o PyTorch foi instalado com CUDA:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Se retornar `False`, reinstale o PyTorch com CUDA (veja passo 2).

**Solução 2:** Verifique os drivers NVIDIA:
```bash
nvidia-smi
```

Se não funcionar, atualize os drivers NVIDIA do site oficial.

**Solução 3:** Verifique a versão do CUDA:
```bash
nvcc --version
```

Certifique-se de que o PyTorch instalado é compatível com sua versão CUDA.

### Problema: "Out of memory"

Se você receber erro de memória, reduza o batch_size:
```bash
python train.py --batch_size 2 --train_dir Healthy_Train50
```

### Problema: GPU não é detectada

1. Verifique se a GPU aparece no Device Manager do Windows
2. Verifique se os drivers NVIDIA estão instalados
3. Reinicie o computador após instalar drivers
4. Execute `nvidia-smi` para verificar se a GPU é reconhecida pelo sistema

## 9. Comparação de Performance

Com GPU RTX 4060, você deve ver:
- **Treinamento:** Muito mais rápido (10-50x mais rápido que CPU)
- **Tempo por época:** Aproximadamente 2-8 minutos (dependendo do batch_size)
- **Tempo total (100 épocas):** Aproximadamente 3-13 horas

Sem GPU (CPU apenas):
- **Tempo por época:** 30-60 minutos ou mais
- **Tempo total (100 épocas):** Muitas horas ou dias

## Resumo Rápido

1. Instale PyTorch com CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. Verifique: `python verificar_gpu.py`
3. Treine: `python train.py --train_dir Healthy_Train50 --epochs 100 --batch_size 4`
4. Monitore: `nvidia-smi -l 1` (em outro terminal)

Pronto! Sua RTX 4060 será usada automaticamente! 🚀

