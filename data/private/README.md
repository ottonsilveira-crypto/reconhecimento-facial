# Dados privados — NÃO PUBLICAR

Para a avaliação privada local, crie uma subpasta por participante que consentiu, por exemplo:

```text
data/private/
├── pessoa_a/
│   ├── foto_01.jpg
│   ├── foto_02.jpg
│   └── ... (3 ou mais fotos)
└── pessoa_b/
    └── ...
```

Use variações de iluminação, expressão, distância e posição da cabeça. Obtenha as imagens diretamente e somente após consentimento informado e específico. O script `python -m src.evaluate_private` divide as fotos de cada pessoa em galeria/cadastro e consultas de teste; calibra o limiar com identidades de validação públicas do LFW e mede também falsos aceites em identidades públicas de teste. Com apenas uma pessoa privada, a taxa de acerto top-1 entre pessoas cadastradas é trivial, portanto interprete os resultados com cuidado.

**Nunca publique** fotografias, recortes faciais, vídeos, termos assinados, embeddings, galerias ou checkpoints derivados de dados privados. A pasta deve permanecer local e protegida. O script não salva fotos nem embeddings e grava apenas métricas agregadas. Não execute em Colab com fotos privadas se as condições de consentimento e as regras institucionais não autorizarem o processamento em nuvem.
