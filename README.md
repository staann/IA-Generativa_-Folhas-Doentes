# Projeto 2 - Diagnóstico de Doenças em Folhas usando IA Generativa

Este projeto implementa um sistema de diagnóstico de doenças em folhas de plantas usando um modelo de IA Generativa (pix2pix) baseado no artigo:

**KATAFUCHI, Ryoya; TOKUNAGA, Terumasa. Image-based plant disease diagnosis with unsupervised anomaly detection based on reconstructability of colors. arXiv preprint arXiv:2011.14306, 2020.**

## Estrutura do Projeto

```
iia_t2/
├── Disease_Test100/          # 100 imagens de folhas doentes para teste
├── Healthy_Test50/           # 50 imagens de folhas saudáveis para teste
├── Healthy_Train50/          # 50 imagens de folhas saudáveis para treinamento
├── models.py                 # Modelos pix2pix (Generator e Discriminator)
├── dataset.py                # Dataset para carregar imagens
├── train.py                  # Script de treinamento
├── test.py                   # Script de teste e avaliação
├── anomaly_detection.py      # Detecção de anomalias baseada em CRI
├── gradcam.py                # Visualização Grad-CAM
├── requirements.txt          # Dependências do projeto
└── README.md                 # Este arquivo
```

## Instalação

1. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Treinamento

Para treinar o modelo pix2pix com as folhas saudáveis:

```bash
python train.py --train_dir Healthy_Train50 --epochs 100 --batch_size 4
```

Parâmetros:
- `--train_dir`: Diretório com imagens de treinamento (padrão: Healthy_Train50)
- `--epochs`: Número de épocas (padrão: 100)
- `--batch_size`: Tamanho do batch (padrão: 4)
- `--lr`: Taxa de aprendizado (padrão: 0.0002)
- `--save_dir`: Diretório para salvar checkpoints (padrão: checkpoints)

### Teste e Avaliação

Para testar o modelo treinado:

```bash
python test.py --generator_path checkpoints/generator_final.pth --healthy_test_dir Healthy_Test50 --disease_test_dir Disease_Test100
```

Parâmetros:
- `--generator_path`: Caminho para o modelo gerador treinado
- `--healthy_test_dir`: Diretório com folhas saudáveis de teste
- `--disease_test_dir`: Diretório com folhas doentes de teste
- `--threshold`: Threshold para classificação de anomalia (padrão: 0.1)
- `--output_dir`: Diretório para salvar resultados (padrão: results)

## Método

O método implementado consiste em:

1. **Treinamento do pix2pix**: O modelo é treinado apenas com folhas saudáveis, aprendendo a reconstruir folhas saudáveis.

2. **Detecção de Anomalias**: Quando uma folha doente é apresentada ao modelo, ele não consegue reconstruí-la adequadamente. O **Índice de Reconstruibilidade de Cores (CRI)** é calculado comparando a imagem original com a reconstruída.

3. **Classificação**: Folhas com CRI acima de um threshold são classificadas como doentes.

4. **Visualização**: 
   - Mapas de anomalias mostram onde o modelo detectou problemas
   - Grad-CAM fornece explicações visuais das decisões do modelo

## Resultados

Os resultados são salvos no diretório `results/` e incluem:
- Métricas de avaliação (acurácia, precisão, recall, F1-score)
- Matriz de confusão
- Distribuição do CRI
- Visualizações de anomalias
- Visualizações Grad-CAM

## Referências

- KATAFUCHI, Ryoya; TOKUNAGA, Terumasa. Image-based plant disease diagnosis with unsupervised anomaly detection based on reconstructability of colors. arXiv preprint arXiv:2011.14306, 2020.
- SELVARAJU, Ramprasaath R. et al. Grad-cam: Visual explanations from deep networks via gradient-based localization. In: Proceedings of the IEEE international conference on computer vision. 2017. p. 618-626.

