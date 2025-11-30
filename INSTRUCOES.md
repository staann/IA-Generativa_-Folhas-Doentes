# Instruções de Uso do Projeto

## Passo a Passo para Executar o Projeto

### 1. Instalação das Dependências

**IMPORTANTE:** Para usar GPU NVIDIA (recomendado), instale PyTorch com CUDA primeiro:

```bash
# Para RTX 4060, use CUDA 11.8 ou 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Depois instale as outras dependências:

```bash
pip install -r requirements.txt
```

**Verificar GPU:**
```bash
python verificar_gpu.py
```

Para mais detalhes sobre GPU, veja `GUIA_GPU.md`

### 2. Treinamento do Modelo

Treine o modelo pix2pix com as folhas saudáveis:

```bash
python train.py --train_dir Healthy_Train50 --epochs 100 --batch_size 4
```

**Parâmetros importantes:**
- `--epochs`: Número de épocas (recomendado: 100-200)
- `--batch_size`: Tamanho do batch (ajuste conforme sua GPU/RAM)
- `--lr`: Taxa de aprendizado (padrão: 0.0002)

**Tempo estimado:** Depende do hardware, mas pode levar várias horas em CPU.

### 3. Teste e Avaliação

Após o treinamento, execute o teste:

```bash
python test.py --generator_path checkpoints/generator_final.pth --healthy_test_dir Healthy_Test50 --disease_test_dir Disease_Test100 --threshold 0.1
```

**Parâmetros importantes:**
- `--threshold`: Threshold para classificação (ajuste conforme necessário)
- Os resultados serão salvos em `results/`

### 4. Execução Completa (Opcional)

Para executar treinamento e teste em sequência:

```bash
python run_project.py --mode both --epochs 100 --batch_size 4
```

## Estrutura de Arquivos Gerados

Após a execução, você terá:

```
checkpoints/
  ├── checkpoint_epoch_10.pth
  ├── checkpoint_epoch_20.pth
  ├── ...
  ├── generator_final.pth
  └── discriminator_final.pth

results/
  ├── results.txt              # Métricas numéricas
  ├── cri_distribution.png     # Gráfico de distribuição do CRI
  ├── visualizations/          # Visualizações de anomalias
  │   ├── healthy_0.png
  │   ├── healthy_1.png
  │   ├── disease_0.png
  │   └── ...
  └── gradcam/                 # Visualizações Grad-CAM
      ├── healthy_0_gradcam.png
      ├── disease_0_gradcam.png
      └── ...
```

## Ajuste do Threshold

O threshold é crucial para a classificação. Para encontrar o melhor valor:

1. Execute o teste com diferentes thresholds
2. Analise a distribuição do CRI em `results/cri_distribution.png`
3. Escolha um threshold que maximize a separação entre classes

## Interpretação dos Resultados

- **CRI baixo (< threshold)**: Folha provavelmente saudável
- **CRI alto (> threshold)**: Folha provavelmente doente
- **Mapas de anomalias**: Mostram onde o modelo detectou problemas
- **Grad-CAM**: Mostra quais áreas da imagem são mais importantes para a decisão

## Troubleshooting

### Erro de memória durante treinamento
- Reduza o `batch_size` (ex: `--batch_size 2` ou `1`)
- Reduza o tamanho da imagem no `dataset.py` (ex: 128x128)

### Modelo não converge
- Aumente o número de épocas
- Ajuste a taxa de aprendizado
- Verifique se as imagens estão sendo carregadas corretamente

### Grad-CAM não funciona
- Isso é normal se houver problemas com os hooks do PyTorch
- O teste continuará sem Grad-CAM, apenas sem as visualizações

## Notas Importantes

1. O treinamento pode demorar muito tempo, especialmente em CPU
2. Recomenda-se usar GPU (CUDA) se disponível
3. Os checkpoints são salvos a cada 10 épocas
4. Você pode continuar o treinamento a partir de um checkpoint (requer modificação do código)

