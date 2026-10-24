# Análise dos resultados — E01, E02 e E03

## Protocolo

- Dataset público: Labeled Faces in the Wild (LFW), carregado via scikit-learn.
- Divisão por identidade: treino (70%), validação (15%) e teste (15%), com seed 42.
- Partições reportadas: treino com 2.238 imagens/110 identidades; validação com 448 imagens/23 identidades; teste com 531 imagens/25 identidades.
- O limiar é selecionado na validação a partir do percentil 99 das similaridades impostoras e então aplicado ao teste.
- O conjunto de teste tem 140.715 pares, gerados a partir das 531 imagens.

## Comparação

| Experimento | Limiar | Accuracy | Precision | Recall | F1 | ROC-AUC | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E01 — baseline | 0,9011 | 0,9464 | 0,2900 | 0,1319 | 0,1813 | 0,7649 | 0,0152 | 0,8681 |
| E02 — cabeça ArcFace | 0,6187 | 0,9461 | 0,3064 | 0,1549 | 0,2058 | 0,8045 | 0,01654 | 0,8451 |
| E03 — fine-tuning parcial | 0,4007 | 0,9504 | 0,3944 | 0,1882 | 0,2548 | 0,8303 | 0,01363 | 0,81177 |

## Interpretação

- E03 teve os melhores valores entre as três configurações em accuracy, precision, recall, F1 e ROC-AUC, além do menor FAR e FRR reportados.
- O aumento de ROC-AUC de 0,7649 (E01) para 0,8303 (E03) sugere melhora na capacidade de ordenar pares genuínos acima de pares impostores, independentemente de um limiar específico.
- Mesmo assim, no limiar escolhido, E03 teve recall de 18,82% e FRR de 81,18%: muitos pares genuínos foram rejeitados. O sistema não deve ser descrito como pronto para uso real.
- Accuracy perto de 95% é potencialmente enganosa porque há muito mais pares impostores do que genuínos. O teste tem 140.715 pares; a matriz de confusão de E03 contém 1.193 TP e 5.145 FN, comparados a 132.545 TN e 1.832 FP.
- O limiar de E03 foi selecionado na validação para mirar FAR próximo de 1%; o FAR de teste observado foi 1,363%. A diferença é esperada quando a distribuição de validação não coincide perfeitamente com a do teste.
- Os limiares diferem entre experimentos e não devem ser comparados como se fossem uma escala de qualidade comum: cada representação produz sua própria distribuição de similaridade.
- A loss de treino cai ao longo de cada fase treinada. Na transição E02→E03, a loss sobe de 16,3682 para 18,2138 porque começa uma nova fase de otimização; depois cai até 13,3504. A loss de treino, isoladamente, não prova generalização.

## Limitações

- LFW é um benchmark histórico e não representa todas as populações nem condições de captura.
- O protocolo por identidade é importante para avaliar identidades não vistas, mas os números ainda dependem da amostra e do procedimento de formação de pares.
- Os pares não são independentes entre si, pois várias combinações reutilizam as mesmas imagens; isso limita interpretações ingênuas de incerteza estatística.
- A ResNet-18 pré-treinada no ImageNet não foi originalmente especializada em reconhecimento facial.
- FAR/FRR dependem do limiar e da distribuição dos dados. O resultado não deve ser usado para vigilância, decisões reais ou identificação sem consentimento.

## Próximas verificações antes da entrega

- Conferir que o gráfico `similarity_distributions.png` corresponde à mesma execução e aos mesmos resultados da tabela.
- Anotar as versões exatas das bibliotecas e o hardware usado (execução reportada: GPU Tesla T4).
- Incluir a demonstração em vídeo e revisar os requisitos completos da atividade.
- Se a avaliação privada fizer parte da entrega, realizá-la apenas com consentimento informado e sem publicar fotos, embeddings, formulários ou checkpoints derivados de dados privados.
