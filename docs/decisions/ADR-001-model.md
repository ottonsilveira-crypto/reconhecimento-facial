# ADR-001 — ResNet-18 + ArcFace como configuração experimental

- **Status:** proposta inicial, sujeita aos resultados.
- **Contexto:** prazo curto; o material didático do professor já demonstra ResNet-18, embeddings, ArcFace e fine-tuning parcial em PyTorch.
- **Decisão:** usar ResNet-18 pré-treinada no ImageNet como baseline; treinar uma cabeça de 128 dimensões com ArcFace e depois liberar `layer4`.
- **Alternativas consideradas:** implementar uma rede do zero; usar uma rede facial especializada com pesos e licença ainda por verificar.
- **Consequências:** implementação relativamente simples e reproduzível; o baseline ImageNet não é um modelo especializado em faces, e o desempenho deve ser avaliado empiricamente.
- **Privacidade/licença:** verificar termos do dataset e dos pesos antes de redistribuir dados ou checkpoints. O repositório não inclui dados nem pesos.
