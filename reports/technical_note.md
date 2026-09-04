# Nota técnica — análise do dataset e plano de solução

## 1. Diagnóstico
O ficheiro recebido é um CSV separado por ponto e vírgula (`;`) com as colunas `ID`, `Text` e `Label`.
Foram encontrados 100 exemplos. Os rótulos aparecem como `Human`, `AI` e uma ocorrência de `Ai`; esta última foi normalizada para `ai`.

Distribuição final:
- Human: 51
- AI: 49

Não foram encontrados textos duplicados. O número de palavras varia de 100 a 123, com média de 114.24. 

## 2. Implicações
O dataset é quase perfeitamente balanceado e os textos têm comprimento próximo do intervalo externo indicado pelo docente. Porém, 100 exemplos são poucos para treinar modelos profundos robustos. Os resultados deste ficheiro devem ser apresentados como experimento/pipeline, não como prova de desempenho geral.

## 3. Estratégia
1. Normalização e controlo de qualidade.
2. Divisão estratificada 70/15/15.
3. TF-IDF implementado com NumPy.
4. Baseline de Logistic Regression implementado de raiz.
5. DNN NumPy com ReLU, Dropout, L2 e Early Stopping.
6. PyTorch DNN, LSTM e Transformer Encoder.
7. Accuracy, Precision, Recall, F1 e matriz de confusão.
8. Seleção final por validação; teste apenas para avaliação final.
9. Validação externa fornecida pelo docente sem alterar o treino.

## 4. Limitação crítica
Com apenas 100 exemplos, um Transformer/LSTM pode memorizar o conjunto. A versão final do trabalho deve ampliar o conjunto de treino usando as fontes autorizadas no enunciado, mantendo separação por fonte quando possível para testar generalização.
