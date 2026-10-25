# Sistema Experimental de Verificação Facial

Projeto acadêmico da disciplina de Inteligência Artificial. A Parte 1 compara três estratégias para verificação facial utilizando embeddings e o dataset público **Labeled Faces in the Wild (LFW)**.

## Pergunta experimental

O treinamento da cabeça de embeddings e o fine-tuning parcial de uma ResNet-18 pré-treinada no ImageNet melhoram o desempenho da verificação facial em identidades não vistas durante o treinamento?

## Experimentos

- **E01 — Baseline:** utiliza uma ResNet-18 pré-treinada no ImageNet, sem fine-tuning.
- **E02 — Cabeça ArcFace:** treina uma cabeça de embeddings de 128 dimensões com ArcFace, mantendo o backbone congelado.
- **E03 — Fine-tuning parcial:** continua o treinamento da cabeça e libera a camada `layer4` da ResNet-18 para ajuste.

As identidades são separadas entre treino, validação e teste, evitando que uma mesma identidade apareça em mais de uma dessas partições. O limiar de decisão é selecionado com base nas similaridades dos pares impostores da validação, utilizando o percentil 99, e depois aplicado ao conjunto de teste.

## Resultados

| Experimento | Accuracy | Precision | Recall | F1-score | ROC-AUC | FAR | FRR |
|---|---:|---:|---:|---:|---:|---:|---:|
| E01 — Baseline | 0,9464 | 0,2900 | 0,1319 | 0,1813 | 0,7649 | 0,0152 | 0,8681 |
| E02 — Cabeça ArcFace | 0,9461 | 0,3064 | 0,1549 | 0,2058 | 0,8045 | 0,01654 | 0,8451 |
| E03 — Fine-tuning parcial | 0,9504 | 0,3944 | 0,1882 | 0,2548 | 0,8303 | 0,01363 | 0,81177 |

Os valores de FAR e FRR correspondem aos resultados observados no conjunto de teste, com os limiares selecionados na validação.

### Interpretação inicial

Entre as três configurações avaliadas, o **E03 apresentou o maior ROC-AUC (0,8303) e F1-score (0,2548)**, além da maior acurácia e da menor FAR e FRR observadas entre os experimentos.

Entretanto, a FRR de aproximadamente 81,18% indica que muitos pares genuínos — imagens da mesma pessoa — ainda são classificados como diferentes no limiar escolhido. O recall de 18,82% reforça essa limitação operacional.

A acurácia, isoladamente, não representa adequadamente o desempenho: o conjunto de teste contém 140.715 pares, dos quais apenas 6.338 são genuínos, enquanto 134.377 são impostores. Portanto, a maioria dos pares pertence à classe negativa, o que pode produzir uma acurácia elevada mesmo quando muitos pares genuínos são rejeitados.

Os valores absolutos de similaridade e os limiares também não devem ser comparados diretamente entre experimentos como se estivessem na mesma escala. A comparação deve considerar as métricas de avaliação e o protocolo utilizado.

Consulte [`docs/analysis-results.md`](docs/analysis-results.md) para a análise detalhada, as limitações e as decisões metodológicas.

### Gráficos e arquivos de resultados

- [`results/metrics.csv`](results/metrics.csv): métricas dos três experimentos.
- [`results/experiment_summary.json`](results/experiment_summary.json): resumo da configuração e dos resultados.
- [`results/training_loss.csv`](results/training_loss.csv): evolução da loss durante o treinamento.
- [`results/similarity_distributions.png`](results/similarity_distributions.png): distribuições de similaridade dos pares genuínos e impostores.

## Como executar

Os experimentos foram executados no Google Colab com GPU NVIDIA Tesla T4.

A partir da raiz do repositório, instale as dependências:

```bash
pip install -r requirements.txt
```

Em seguida, execute os experimentos:

```bash
python -m src.run_experiments
```

O primeiro acesso ao LFW requer conexão com a internet. O dataset é obtido por meio do scikit-learn e armazenado em cache local. Não publique o dataset ou seu cache no repositório.

Para reproduzir os resultados, utilize as configurações e a semente aleatória registradas no projeto. Diferenças de versões de bibliotecas, ambiente e hardware podem afetar a reprodução exata.

## Estrutura do repositório

```text
configs/       Configurações dos experimentos
data/private/  Local reservado para dados privados, não versionados
docs/          Especificações, decisões e análise dos resultados
results/       Métricas, resumo, loss e gráficos
src/           Código dos experimentos e avaliação
tests/         Testes automatizados
```

## Privacidade e uso responsável

Não publique fotografias privadas, recortes faciais, vídeos, embeddings, galerias biométricas, termos de consentimento ou checkpoints derivados de dados privados.

O `.gitignore` ajuda a excluir arquivos locais e checkpoints, mas não substitui uma revisão manual dos arquivos enviados. Antes de publicar, confira o conteúdo do repositório e as condições de uso dos datasets, pesos pré-treinados e dependências.

A coleta e a avaliação com participantes devem ocorrer somente com consentimento informado e específico, respeitando as condições aplicáveis ao tratamento de dados biométricos. Este projeto é experimental e não deve ser utilizado para vigilância ou decisões reais.

## Referência

A arquitetura experimental toma como referência o material de aula sobre Metric Learning, com adaptação do protocolo para separar as identidades de treino, validação e teste. As escolhas metodológicas e limitações são discutidas no relatório de análise.
