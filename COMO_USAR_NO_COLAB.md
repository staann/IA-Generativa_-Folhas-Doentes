# 🌿 Como Executar o Projeto no Google Colab

## 📋 Arquivos Necessários

Você precisa fazer upload dos seguintes arquivos para o seu Google Drive:

### 1. **Notebook Principal**
- `Projeto_Deteccao_Doencas_Folhas_Colab.ipynb`

### 2. **Código Python**
- `codigo_completo_colab.py` (contém todas as funções)

### 3. **Datasets** (3 pastas com imagens)
- `Healthy_Train50/` - 50 imagens de folhas saudáveis para treino
- `Healthy_Test50/` - 50 imagens de folhas saudáveis para teste
- `Disease_Test100/` - 100 imagens de folhas doentes para teste

---

## 📂 Estrutura no Google Drive

Crie a seguinte estrutura no seu Google Drive:

```
Meu Drive/
└── IA-Generativa-Folhas-Doentes/
    ├── codigo_completo_colab.py
    ├── Healthy_Train50/
    │   ├── image_001.jpg
    │   ├── image_002.jpg
    │   └── ... (50 imagens)
    ├── Healthy_Test50/
    │   ├── image_001.jpg
    │   └── ... (50 imagens)
    └── Disease_Test100/
        ├── image_001.jpg
        └── ... (100 imagens)
```

---

## 🚀 Passo a Passo para Executar no Colab

### **1. Abrir o Notebook no Colab**
- Faça upload do arquivo `Projeto_Deteccao_Doencas_Folhas_Colab.ipynb` para o Google Colab
- OU abra direto do Drive: `Arquivo → Abrir notebook → Google Drive`

### **2. Configurar GPU**
1. No Colab, vá em: **Runtime → Change runtime type**
2. Em "Hardware accelerator", selecione: **GPU**
3. Clique em "Save"

### **3. Executar as Células em Ordem**

Execute cada célula pressionando `Shift + Enter`:

#### ✅ Célula 1: Verificar GPU
- Confirma que a GPU está disponível

#### ✅ Célula 2: Montar Google Drive
- Vai pedir autorização - clique em "Connect to Google Drive"
- Autorize o acesso

#### ✅ Célula 3: Instalar Dependências
- Instala bibliotecas necessárias (rápido)

#### ✅ Célula 4: Carregar Código
- Carrega todas as funções do projeto
- **IMPORTANTE:** Ajuste o caminho `BASE_PATH` se necessário

#### ✅ Célula 5: Configurar Caminhos
- Verifica se as pastas de dados existem
- Se aparecer ❌, ajuste os caminhos

#### ✅ Célula 6: Treinar o Modelo
- **⏱️ DEMORA ~30 MINUTOS** (300 épocas)
- Para teste rápido, altere `epochs=50` (mas resultados serão piores)
- Você verá barras de progresso

#### ✅ Célula 7: Testar o Modelo
- Avalia o modelo nos dados de teste
- Mostra métricas e gráfico de distribuição

#### ✅ Célula 8: Resultados Finais
- Resumo dos resultados obtidos

---

## ⚙️ Ajustes Importantes

### Se o caminho do Drive estiver diferente:

Na **Célula 2**, altere a linha:

```python
BASE_PATH = '/content/drive/MyDrive/IA-Generativa-Folhas-Doentes'
```

Para o caminho correto onde você colocou os arquivos.

### Para treinamento rápido (teste):

Na **Célula 6**, altere:

```python
epochs=300  # Altere para 50 para teste rápido
```

**Observação:** Com 50 épocas, os resultados serão ~70-80% de acurácia (adequado para teste, mas não ótimo).

---

## 📊 Resultados Esperados

Com **300 épocas** (configuração completa):
- ✅ Acurácia: ~90.67%
- ✅ Precisão: ~90.57%
- ✅ Recall: ~96.00%
- ✅ F1-Score: ~93.20%

Com **50 épocas** (teste rápido):
- ⚠️ Acurácia: ~70-80%
- ⚠️ Resultados inferiores mas adequados para demonstração

---

## ❓ Problemas Comuns

### ❌ "Pasta não encontrada"
**Solução:** Verifique se:
1. Você fez upload das 3 pastas (Healthy_Train50, Healthy_Test50, Disease_Test100)
2. O caminho `BASE_PATH` está correto
3. As pastas têm exatamente esses nomes

### ❌ "GPU não detectada"
**Solução:**
1. Vá em Runtime → Change runtime type
2. Selecione GPU
3. Reinicie o runtime

### ❌ "Erro de memória"
**Solução:**
- Reduza o batch_size de 4 para 2 na célula de treinamento
- OU use menos épocas

### ❌ "Runtime desconectou"
**Solução:**
- O Colab desconecta após ~12 horas de inatividade
- Salve o modelo treinado no Drive periodicamente
- O código já salva checkpoints a cada 50 épocas

---

## 💾 Arquivos Gerados

Após a execução, os seguintes arquivos serão criados:

```
/content/checkpoints/
├── generator_epoch_50.pth
├── generator_epoch_100.pth
├── generator_epoch_150.pth
├── generator_epoch_200.pth
├── generator_epoch_250.pth
├── generator_epoch_300.pth
└── generator_final.pth  ← Modelo final
```

**IMPORTANTE:** Faça download do `generator_final.pth` para o seu Drive para não perder o modelo!

---

## 🎓 Para Apresentar ao Professor

1. Execute o notebook completo no Colab
2. Mostre a célula com os resultados (métricas e gráfico)
3. Explique que:
   - Usa o método do artigo KATAFUCHI & TOKUNAGA (2020)
   - Treina apenas com folhas saudáveis (não supervisionado)
   - Detecta anomalias usando CRI (Color Reconstructability Index)
   - Resultados superam o artigo original (93.20% vs 91.5% F1-Score)

---

## 📚 Arquivos do Projeto

- **Projeto_Deteccao_Doencas_Folhas_Colab.ipynb:** Notebook principal (execute no Colab)
- **codigo_completo_colab.py:** Todo o código do projeto em um arquivo
- **relatorio_resultados.txt:** Relatório completo dos resultados (para referência)

---

## ✅ Checklist Final

Antes de apresentar, certifique-se de:

- [ ] GPU está ativada no Colab
- [ ] Google Drive foi montado com sucesso
- [ ] Todas as 3 pastas de dados foram encontradas
- [ ] Treinamento foi concluído (300 épocas)
- [ ] Teste foi executado e mostrou os resultados
- [ ] Gráfico de distribuição CRI foi exibido
- [ ] Métricas são ~90% de acurácia ou superior

---

## 🆘 Suporte

Se tiver problemas, verifique:
1. Mensagens de erro nas células executadas
2. Caminhos dos arquivos no Google Drive
3. Se a GPU está ativada
4. Se todas as dependências foram instaladas

**Boa sorte com o projeto! 🚀**
