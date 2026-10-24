# SPEC-001 — Experimentos da Parte 1

## Objetivo
Investigar se o treinamento de uma cabeça de embeddings e o fine-tuning parcial de uma ResNet-18 melhoram a verificação facial em identidades não vistas durante o treinamento.

## Escopo
- Dataset público LFW obtido pela fonte usada pelo scikit-learn.
- ResNet-18 com pesos ImageNet como referência.
- Embeddings de 128 dimensões para o modelo ajustado.
- ArcFace como função de perda.
- Comparação por similaridade de cosseno.
- Threshold selecionado exclusivamente na validação.
- Métricas de pares genuínos/impostores no teste final.

## Fora de escopo desta etapa
- Aplicação final com interface.
- Publicação de fotografias, embeddings, galerias ou modelos derivados de dados privados.
- Alegações de adequação para vigilância ou uso real de alto impacto.

## Critérios de aceitação
1. Seed registrada.
2. Identidades de treino, validação e teste disjuntas.
3. Baseline pré-treinado medido antes de fine-tuning.
4. Pelo menos uma configuração com fine-tuning real da rede.
5. Threshold calculado apenas a partir da validação.
6. Teste utilizado uma única vez para avaliação final.
7. CSV e gráfico gerados com métricas agregadas.
8. Ambiente, versões, hiperparâmetros e limitações documentados.
9. Nenhum dado biométrico privado no repositório público.

## Métricas
Accuracy, Precision, Recall, F1, ROC-AUC, FAR, FRR, médias de similaridade genuína/impostora e matriz de confusão (TN, FP, FN, TP).

## Hipóteses
- H1: o ajuste da cabeça de embeddings com ArcFace separa melhor pares genuínos e impostores que a saída ImageNet sem treinamento.
- H2: liberar `layer4` pode melhorar a discriminação facial, mas pode também sobreajustar ao conjunto de treinamento.
- H3: o threshold escolhido na validação oferece um compromisso entre falsos aceites e falsas rejeições no teste; a taxa real no teste pode divergir do alvo.
