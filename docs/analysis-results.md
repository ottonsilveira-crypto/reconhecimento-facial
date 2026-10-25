# Análise dos resultados — E01, E02 e E03

## 1. Protocolo experimental

- **Dataset:** Labeled Faces in the Wild (LFW), carregado por meio do scikit-learn.
- **Divisão por identidade:** treino (70%), validação (15%) e teste (15%), com semente aleatória (`seed`) 42.
- **Partições reportadas:** treino com 2.238 imagens e 110 identidades; validação com 448 imagens e 23 identidades; teste com 531 imagens e 25 identidades.
- **Seleção do limiar:** o limiar de decisão é definido na validação a partir do percentil 99 das similaridades dos pares impostores, buscando uma taxa de falsa aceitação próxima de 1%. O limiar selecionado é aplicado ao teste.
- **Formação dos pares:** as 531 imagens de teste geram 140.715 pares não ordenados de imagens distintas, correspondentes a todas as combinações de duas imagens. Cada par é classificado como genuíno (mesma identidade) ou impostor (identidades diferentes).

Os pares não são estatisticamente independentes, pois uma mesma imagem participa de múltiplas combinações. Essa característica deve ser considerada na interpretação dos resultados.

## 2. Comparação dos experimentos

| Experimento | Limiar | Accuracy | Precision | Recall | F1-score | ROC-AUC | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 — Baseline | 0,9011 | 0,9464 | 0,2900 | 0,1319 | 0,1813 | 0,7649 | 0,0152 | 0,8681 |
| E02 — Cabeça ArcFace | 0,6187 | 0,9461 | 0,3064 | 0,1549 | 0,2058 | 0,8045 | 0,01654 | 0,8451 |
| E03 — Fine-tuning parcial | 0,4007 | 0,9504 | 0,3944 | 0,1882 | 0,2548 | 0,8303 | 0,01363 | 0,81177 |

**Definições das métricas:**

- **Accuracy:** proporção total de pares classificados corretamente.
- **Precision:** proporção de pares classificados como genuínos que realmente são genuínos.
- **Recall:** proporção de pares genuínos identificados corretamente.
- **F1-score:** média harmônica entre precisão e recall.
- **ROC-AUC:** mede a capacidade de ordenar pares genuínos acima de pares impostores ao longo de diferentes limiares.
- **FAR (False Acceptance Rate):** proporção de pares impostores classificados incorretamente como genuínos.
- **FRR (False Rejection Rate):** proporção de pares genuínos classificados incorretamente como impostores.

## 3. Interpretação dos resultados

### 3.1. Comparação geral

O E03 apresentou os melhores resultados entre as três configurações avaliadas: maior accuracy (0,9504), precision (0,3944), recall (0,1882), F1-score (0,2548) e ROC-AUC (0,8303), além dos menores valores observados de FAR (0,01363) e FRR (0,81177).

O aumento de ROC-AUC de 0,7649 no E01 para 0,8303 no E03 sugere melhora na capacidade de discriminar e ordenar pares genuínos em relação aos impostores, considerando diferentes limiares de decisão.

Os resultados são compatíveis com a hipótese de que treinar a cabeça de embeddings e realizar fine-tuning parcial podem melhorar a verificação facial nesse protocolo experimental. Entretanto, os resultados de um único protocolo e uma única semente não permitem generalizar essa conclusão para todas as populações e condições de captura.

### 3.2. Limitações no limiar escolhido

Apesar da melhora relativa, o E03 apresentou recall de apenas 18,82% e FRR de aproximadamente 81,18% no limiar selecionado. Isso significa que a maior parte dos pares genuínos do teste foi rejeitada nesse ponto de operação.

O resultado evidencia uma limitação importante: melhor discriminação global, indicada pelo ROC-AUC, não implica necessariamente desempenho satisfatório no limiar escolhido. O modelo não deve ser descrito como pronto para uso real.

### 3.3. Desequilíbrio entre pares genuínos e impostores

O conjunto de teste contém 140.715 pares, distribuídos da seguinte forma para o E03:

- **TP (verdadeiros positivos):** 1.193 pares genuínos corretamente identificados.
- **FN (falsos negativos):** 5.145 pares genuínos rejeitados.
- **TN (verdadeiros negativos):** 132.545 pares impostores corretamente rejeitados.
- **FP (falsos positivos):** 1.832 pares impostores aceitos incorretamente.

Assim, existem 6.338 pares genuínos e 134.377 pares impostores. Aproximadamente 95,5% dos pares pertencem à classe impostora.

Esse desequilíbrio ajuda a explicar por que a accuracy fica próxima de 95%, embora o recall dos pares genuínos seja baixo. Por isso, a accuracy não deve ser interpretada isoladamente; precision, recall, F1-score, ROC-AUC, FAR e FRR fornecem informações complementares sobre o comportamento do sistema.

### 3.4. Seleção do limiar e comparação entre modelos

O limiar é selecionado na validação com base no percentil 99 das similaridades impostoras, buscando uma FAR próxima de 1%. No teste, a FAR observada no E03 foi de 1,363%.

A diferença entre a taxa pretendida na validação e a observada no teste é compatível com variações amostrais e diferenças entre as distribuições de similaridade dos dois conjuntos. O procedimento não garante uma FAR de exatamente 1% em dados não vistos.

Os limiares dos experimentos — 0,9011 no E01, 0,6187 no E02 e 0,4007 no E03 — não devem ser comparados como se representassem uma escala comum de qualidade. Cada representação pode produzir distribuições de similaridade e escalas numéricas diferentes. A comparação deve considerar as métricas e o protocolo de avaliação.

### 3.5. Evolução da loss de treinamento

A loss diminuiu ao longo das fases de treinamento:

- **E02 — Cabeça ArcFace:** de 19,1669 na primeira época para 16,3682 na terceira.
- **E03 — Fine-tuning parcial:** de 18,2138 na primeira época para 13,3504 na quinta.

A loss aumenta de 16,3682 no final do E02 para 18,2138 no início do E03. Essa transição corresponde ao início de uma nova fase de otimização, com uma configuração de treinamento diferente, incluindo o desbloqueio parcial do backbone. Portanto, esses valores não representam uma comparação direta sob condições idênticas.

A redução da loss dentro de cada fase indica progresso no objetivo de treinamento, mas, isoladamente, não demonstra generalização. Essa avaliação depende das métricas obtidas nos dados de validação e teste.

## 4. Limitações do estudo

- **Representatividade:** o LFW é um benchmark histórico e não representa todas as populações, condições de iluminação, poses, câmeras ou ambientes de captura.
- **Dependência da amostra:** os resultados dependem das identidades, imagens e do procedimento de formação de pares utilizado.
- **Dependência entre observações:** os pares reutilizam as mesmas imagens, de modo que não podem ser tratados como observações estatisticamente independentes em análises ingênuas de incerteza.
- **Modelo pré-treinado:** a ResNet-18 utilizada foi pré-treinada no ImageNet e não foi originalmente especializada em reconhecimento facial.
- **Limiar operacional:** FAR e FRR dependem do limiar e da distribuição dos dados; alterações na população ou nas condições de captura podem modificar essas taxas.
- **Reprodutibilidade:** versões de bibliotecas, hardware, operações não determinísticas e configurações de treinamento podem afetar a reprodução exata dos resultados.
- **Escopo:** a avaliação realizada não comprova adequação para vigilância, identificação em situações reais ou decisões de alto impacto.

O projeto deve ser considerado experimental. Qualquer avaliação com participantes deve respeitar o consentimento informado, a finalidade autorizada e as condições aplicáveis ao tratamento de dados biométricos.

## 5. Verificações antes da entrega

- [ ] Confirmar que `results/similarity_distributions.png` corresponde à mesma execução que produziu as métricas apresentadas.
- [ ] Confirmar que `results/experiment_summary.json` contém as configurações e os resultados dos três experimentos.
- [ ] Registrar as versões efetivamente utilizadas das bibliotecas e o hardware da execução. O ambiente reportado foi uma GPU NVIDIA Tesla T4 no Google Colab.
- [ ] Conferir a consistência entre este relatório, o `README.md`, os CSVs e o resumo JSON.
- [ ] Preparar a demonstração em vídeo e verificar todos os requisitos da atividade.
- [ ] Se houver avaliação privada na atividade, realizá-la apenas com consentimento adequado e não publicar fotos, embeddings, formulários ou checkpoints derivados de dados privados.
