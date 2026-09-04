## UKB-MEI_ProjetoDL_Grupo_8
Projeto Prático de Deep Learning UKB-MEI/2026 - Detecção de Texto Gerado por IA vs Humano.

INTEGRANTES:
  1. Gonçalves Faria Felizardo - gfelizardo96@gmail.com
  2. Dumilde Filipe Kambango - dumilderaby@gmail.com
  3. Edilson Junqueira C. Canõma - jaguasj@gmail.com

## AI vs Human Text Detection — Trabalho de Aprendizagem Profunda

  1. Enunciado: [Enunciado_Trabalho.pdf](https://github.com/user-attachments/files/31832144/Enunciado_Trabalho.pdf)
  2. DataSet: [dataset_complete.csv](https://github.com/user-attachments/files/31833068/dataset_complete.csv)

## Objetivo
Classificar textos em inglês em duas classes: `human` e `ai`, seguindo o enunciado do trabalho prático.

## Dataset recebido
O ficheiro original contém **100 exemplos**. Após normalização dos rótulos:
- Human: 51
- AI: 49
- Comprimento: 100–123 palavras
- Média: ~114 palavras
- Duplicados de texto: 0

O intervalo de comprimento é especialmente compatível com a descrição da validação externa do docente (100–120 palavras), mas o dataset continua pequeno para conclusões fortes de generalização.

## Conformidade com as regras
### Implementação própria
`src/models.py` implementa com NumPy:
1. TF-IDF
2. Logistic Regression
3. DNN
4. ReLU
5. Backpropagation
6. Dropout
7. L2 regularization
8. Early stopping
9. He initialization

**Não usar scikit-learn, TensorFlow, Keras ou PyTorch nos modelos NumPy.**

### PyTorch
São fornecidos:
1. DNN + TF-IDF
2. LSTM + embeddings
3. Transformer Encoder pequeno treinado de raiz

## Execução
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Execute os notebooks pela ordem:
1. `01_dataset_analysis.ipynb`
2. `02_numpy_models.ipynb`
3. `03_pytorch_models.ipynb`

## Recomendação para a submissão
O dataset recebido é ótimo para validar o pipeline, mas tem apenas 100 exemplos. Para obter melhor generalização e uma avaliação externa mais robusta, combinar fontes autorizadas pelo docente e deduplicar antes da divisão.

## Estrutura
- `data/raw` — dataset original normalizado
- `data/processed` — dataset limpo e partições
- `src/models.py` — algoritmos
- `notebooks` — análise, NumPy e PyTorch
- `models` — guardar pesos finais
- `reports` — análises e resultados

