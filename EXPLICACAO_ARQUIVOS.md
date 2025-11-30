# Explicação dos Arquivos do Projeto

Este documento explica para que serve cada arquivo principal do projeto.

## 📁 models.py

**Função:** Define a arquitetura das redes neurais (modelos) usadas no projeto.

**O que contém:**
- **`UNetDown`**: Blocos de redução (downsampling) do U-Net
- **`UNetUp`**: Blocos de aumento (upsampling) do U-Net
- **`Generator`**: O gerador pix2pix baseado em arquitetura U-Net
  - Recebe uma imagem de folha como entrada
  - Tenta reconstruir/reproduzir a imagem
  - Treinado apenas com folhas saudáveis, então reconstrói bem folhas saudáveis
- **`Discriminator`**: O discriminador PatchGAN
  - Ajuda no treinamento do gerador
  - Diferencia entre imagens reais e geradas

**Como é usado:**
- Importado em `train.py` para criar os modelos
- Importado em `test.py` para carregar o modelo treinado
- O gerador é o componente principal que faz a reconstrução das imagens

**Exemplo de uso:**
```python
from models import Generator, Discriminator
generator = Generator()  # Cria o gerador
discriminator = Discriminator()  # Cria o discriminador
```

---

## 📁 anomaly_detection.py

**Função:** Implementa a detecção de anomalias (folhas doentes) usando o método do artigo.

**O que contém:**
- **`calculate_color_reconstructability_index()`**: 
  - Calcula o **CRI (Color Reconstructability Index)** - Índice de Reconstruibilidade de Cores
  - Compara a imagem original com a reconstruída pelo gerador
  - Quanto maior o CRI, maior a diferença = maior probabilidade de ser doente
  - Retorna o CRI e um mapa de anomalias mostrando onde estão as diferenças

- **`detect_anomaly()`**: 
  - Função principal para detectar se uma folha está doente
  - Carrega a imagem, passa pelo gerador, calcula o CRI
  - Classifica como doente se CRI > threshold

- **`visualize_anomaly()`**: 
  - Cria visualizações mostrando:
    - Imagem original
    - Imagem reconstruída
    - Mapa de anomalias (onde o modelo detectou problemas)
    - Sobreposição do mapa na imagem original

**Como é usado:**
- Importado em `test.py` para avaliar as imagens de teste
- Usado para classificar folhas como saudáveis ou doentes
- Gera as visualizações que são salvas em `results/visualizations/`

**Exemplo de uso:**
```python
from anomaly_detection import detect_anomaly
is_doente, cri, mapa, reconstruida = detect_anomaly(
    "imagem.jpg", generator, device, threshold=0.1
)
```

**Lógica:**
1. Modelo treinado apenas com folhas saudáveis
2. Quando recebe folha saudável → reconstrói bem → CRI baixo
3. Quando recebe folha doente → não reconstrói bem → CRI alto
4. Se CRI > threshold → classifica como doente

---

## 📁 gradcam.py

**Função:** Implementa Grad-CAM para explicar visualmente quais partes da imagem o modelo considera mais importantes.

**O que contém:**
- **`GradCAM`**: Classe que implementa o algoritmo Grad-CAM
  - Captura gradientes e ativações de uma camada específica do modelo
  - Gera um mapa de calor mostrando onde o modelo "olha" ao tomar decisões
  - Áreas mais quentes (vermelhas) = mais importantes para a decisão

- **`create_gradcam_for_generator()`**: 
  - Função auxiliar para criar instância de GradCAM para o gerador
  - Seleciona automaticamente uma camada apropriada

**Como é usado:**
- Importado em `test.py` para gerar explicações visuais
- Cria imagens mostrando quais partes da folha o modelo considera importantes
- Salva visualizações em `results/gradcam/`

**Por que é importante:**
- **Transparência**: Mostra por que o modelo classificou como doente/saudável
- **Validação**: Permite verificar se o modelo está olhando para as áreas certas
- **Debugging**: Ajuda a identificar problemas no modelo

**Exemplo de uso:**
```python
from gradcam import create_gradcam_for_generator
gradcam = create_gradcam_for_generator(generator, device)
gradcam.visualize(imagem, reconstruida, caminho_original, "resultado.png")
```

**O que mostra:**
- Imagem original
- Mapa Grad-CAM (mapa de calor)
- Sobreposição do Grad-CAM na imagem original

---

## 🔄 Como os Arquivos Trabalham Juntos

```
┌─────────────┐
│  models.py  │  Define a arquitetura do modelo
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  train.py   │  Usa models.py para criar e treinar o modelo
└──────┬──────┘
       │
       ▼ (modelo treinado)
┌─────────────┐
│  test.py    │  Carrega modelo treinado
└──────┬──────┘
       │
       ├──► anomaly_detection.py  ──► Detecta se folha está doente
       │                              Calcula CRI
       │                              Gera mapas de anomalias
       │
       └──► gradcam.py  ──► Explica visualmente a decisão
                            Mostra onde o modelo "olha"
                            Gera mapas de calor
```

## 📊 Fluxo Completo

1. **Treinamento** (`train.py`):
   - Usa `models.py` para criar Generator e Discriminator
   - Treina apenas com folhas saudáveis
   - Salva modelo treinado

2. **Teste** (`test.py`):
   - Carrega modelo de `models.py`
   - Para cada imagem de teste:
     - Usa `anomaly_detection.py` para:
       - Passar imagem pelo gerador
       - Calcular CRI
       - Classificar como saudável/doente
       - Gerar visualizações
     - Usa `gradcam.py` para:
       - Gerar explicações visuais
       - Mostrar áreas importantes

3. **Resultados**:
   - Métricas (acurácia, precisão, etc.)
   - Visualizações de anomalias
   - Visualizações Grad-CAM

## 🎯 Resumo Rápido

| Arquivo | Função Principal |
|---------|------------------|
| **models.py** | Define as redes neurais (Generator e Discriminator) |
| **anomaly_detection.py** | Detecta folhas doentes calculando o CRI |
| **gradcam.py** | Explica visualmente as decisões do modelo |

Todos trabalham juntos para:
1. Treinar um modelo que reconstrói folhas saudáveis
2. Detectar folhas doentes pela diferença na reconstrução
3. Explicar visualmente as decisões do modelo

