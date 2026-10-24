# Sistema Experimental de Reconhecimento Facial

Projeto acadêmico da disciplina de Inteligência Artificial. A Parte 1 compara três estratégias para verificação facial usando embeddings e o dataset público Labeled Faces in the Wild (LFW).

## Pergunta experimental

O treinamento da cabeça de embeddings e o fine-tuning parcial de uma ResNet-18 pré-treinada no ImageNet melhoram a verificação facial em identidades não vistas durante o treinamento?

## Experimentos

- **E01 — Baseline:** ResNet-18 pré-treinada no ImageNet, sem fine-tuning.
- **E02 — Cabeça ArcFace:** treina uma cabeça de embeddings de 128 dimensões, mantendo o backbone congelado.
- **E03 — Fine-tuning parcial:** treina a cabeça e libera a camada `layer4` da ResNet-18.

As identidades são separadas entre treino, validação e teste. O limiar é selecionado com a validação e aplicado ao teste.

## Resultados resumidos

| Experimento | Accuracy | F1 | ROC-AUC | FAR | FRR |
|---|---:|---:|---:|---:|---:|
| E01 | 0,9464 | 0,1813 | 0,7649 | 0,0152 | 0,8681 |
| E02 | 0,9461 | 0,2058 | 0,8045 | 0,01654 | 0,8451 |
| E03 | 0,9504 | 0,2548 | 0,8303 | 0,01363 | 0,81177 |

E03 obteve o melhor resultado geral entre as três configurações avaliadas, mas a FRR de 81,18% mostra que muitos pares genuínos ainda são rejeitados no limiar escolhido. Consulte [`docs/analysis-results.md`](docs/analysis-results.md) para o protocolo, métricas completas, interpretação e limitações.

## Como executar

Ambiente testado no Google Colab com GPU Tesla T4. As versões exatas podem variar; registre as versões efetivamente usadas na execução final.

```bash
pip install -r requirements.txt
python -m src.run_experiments
```

O primeiro acesso ao LFW requer conexão com a internet. O dataset é baixado/carregado pelo scikit-learn e mantido em cache local; não envie o dataset ou o cache ao GitHub.

## Estrutura do repositório

```text
configs/       Configuração dos experimentos
data/private/  Apenas instruções; dados privados não são versionados
docs/          Especificação, decisão de arquitetura e análise dos resultados
results/       Métricas agregadas e evolução da loss
src/           Código dos experimentos e avaliação
tests/        Testes automatizados
```

## Privacidade e arquivos que não devem ser publicados

Não publique fotos privadas, recortes faciais, vídeos, embeddings, galerias biométricas, termos de consentimento ou checkpoints derivados de dados privados. O `.gitignore` exclui checkpoints e dados locais por padrão. Antes de publicar, confira `git status` e revise a licença dos datasets, pesos pré-treinados e dependências.

A coleta e a avaliação com pessoas devem ser feitas apenas com consentimento informado e específico. Não use o sistema para vigilância ou decisões reais.

## Referência

A arquitetura experimental toma como referência o material de aula de Metric Learning, com adaptação do protocolo para separar as identidades de treino, validação e teste. Consulte o relatório para as decisões e limitações.
