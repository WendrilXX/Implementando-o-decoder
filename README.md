# Laboratório 2: Implementando o Decoder do Transformer

## Estrutura do Projeto

Este laboratório contém **3 tarefas independentes** separadas em arquivos distintos:

```
├── main.py                          # Coordenador (executa todas as tarefas)
├── decoder_laboratory.py            # Versão monolítica (todas tarefas em 1 arquivo)
├── tarefa1_mascara_causal.py        # TAREFA 1: Máscara Causal
├── tarefa2_cross_attention.py       # TAREFA 2: Cross-Attention
└── tarefa3_loop_autoregressive.py   # TAREFA 3: Loop Auto-Regressivo
```

## Objetivos de Aprendizagem

1. **Dominar a álgebra linear do mascaramento causal** (Look-Ahead Masking)
2. **Implementar a ponte Encoder-Decoder** (Cross-Attention)
3. **Simular um loop de geração auto-regressiva**

---

## Como Executar

### Opção 1: Executar TODAS as tarefas em sequência
```bash
python main.py
```

### Opção 2: Executar uma tarefa específica
```bash
python tarefa1_mascara_causal.py
python tarefa2_cross_attention.py
python tarefa3_loop_autoregressive.py
```

### Opção 3: Executar versão monolítica
```bash
python decoder_laboratory.py
```

---

## Descrição Detalhada de Cada Tarefa

### TAREFA 1: Máscara Causal (Look-Ahead Mask)

**Arquivo:** `tarefa1_mascara_causal.py`

**Conceito:**
No treinamento paralelizado em GPUs, a frase de destino completa entra no Decoder. Para impedir que a palavra na posição `i` atenda à posição `i+1`, injetamos uma máscara matricial `M` antes do cálculo do Softmax.

**O que foi implementado:**
- Função `create_causal_mask(seq_len)` 
- Função `softmax()` com estabilidade numérica
- Prova Real: Multiplica Q e K, adiciona máscara, aplica Softmax
- Verificação: Posições futuras têm probabilidade **exatamente 0.0**

**Matemática:**
```
Máscara para seq_len=5:
[[  0  -∞  -∞  -∞  -∞]
 [  0   0  -∞  -∞  -∞]
 [  0   0   0  -∞  -∞]
 [  0   0   0   0  -∞]
 [  0   0   0   0   0]]

Scores masked:
scores + mask

Softmax:
attention_weights = softmax(scores_masked)
```

**Resultado Esperado:**
```
Posição 0: [1.0, 0.0, 0.0, 0.0, 0.0] - Só vê a si mesmo
Posição 1: [X, X, 0.0, 0.0, 0.0] - Vê posições 0-1
Posição 2: [X, X, X, 0.0, 0.0] - Vê posições 0-2
...
```

---

### TAREFA 2: Cross-Attention (Ponte Encoder-Decoder)

**Arquivo:** `tarefa2_cross_attention.py`

**Conceito:**
Diferente do Self-Attention, onde as representações derivam do mesmo texto, o Encoder-Decoder Attention cruza fronteiras. O estado atual da geração interage com a "memória" lida pelo Encoder.

**O que foi implementado:**
- Tensores fictícios:
  - `encoder_output` [batch=1, seq_len_francês=10, d_model=512]
  - `decoder_state` [batch=1, seq_len_inglês=4, d_model=512]
- Função `cross_attention(encoder_out, decoder_state)`
- Projeção de decoder_state → Query (Q)
- Projeção de encoder_out → Keys (K) e Values (V)
- Cálculo de Scaled Dot-Product Attention
- Softmax **SEM máscara causal** (pode olhar encoder por completo)

**Fluxo:**
```
decoder_state ──→ W_q ──→ Q [batch, seq_decoder, d_k]
                                    ↓
encoder_output ──→ W_k ──→ K [batch, seq_encoder, d_k]
                ↓
                W_v ──→ V [batch, seq_encoder, d_v]

scores = (Q @ K^T) / sqrt(d_k)  [batch, seq_decoder, seq_encoder]
weights = Softmax(scores)
output = weights @ V            [batch, seq_decoder, d_v]
```

**Resultado Esperado:**
```
Cada posição decoder: distribui atenção sobre 10 posições encoder
Soma dos pesos: 1.0 (distribuição de probabilidade)
Sem bloqueios (máscara causal): pode acessar qualquer posição do encoder
```

---

### TAREFA 3: Loop de Inferência Auto-Regressivo

**Arquivo:** `tarefa3_loop_autoregressive.py`

**Conceito:**
O modelo de linguagem trabalha como um laço de repetição condicionado na produção anterior. Após as sub-camadas, o vetor final de 512 dimensões sofre uma projeção linear para o tamanho do vocabulário, seguida de um Softmax.

**O que foi implementado:**
- Classe `MockVocabulary` (10.000 tokens fictícios)
  - Tokens especiais: `<START>`, `<EOS>`, `<UNK>`
  - Tokens regulares: `word_0`, `word_1`, ..., `word_9999`
- Função `generate_next_token(current_sequence, encoder_out)`
  - Self-Attention com máscara causal
  - Cross-Attention com encoder
  - FFN (feed-forward)
  - Output projection → vocabulário
  - Retorna distribuição de probabilidades
- Função `generate_with_argmax()` (loop while)
  - Começa com `<START>`
  - A cada passo: aplica argmax para token mais provável
  - Adiciona à sequência
  - **Para se gerar `<EOS>` ou atingir `max_length`**

**Pipeline de Geração:**
```
Passo 1:  <START> 
          ↓
          [Self-Attention com máscara causal]
          [Cross-Attention com encoder]
          [FFN]
          ↓
          Distribuição P(token_1 | <START>, encoder)
          argmax → word_4023
          
Passo 2:  <START> → word_4023
          ↓
          [Self-Attention (vê <START> e word_4023)]
          [Cross-Attention com encoder]
          [FFN]
          ↓
          Distribuição P(token_2 | <START> word_4023, encoder)
          argmax → word_7460
          
... (continua até <EOS> ou max_length)
```

**Resultado Esperado:**
```
Iteração 1: <START> → word_4023 (prob: 0.3820)
Iteração 2: word_4023 → word_7460 (prob: 0.4354)
Iteração 3: word_7460 → word_3429 (prob: 0.6860)
...
Iteração N: token → <EOS> (prob: X)
Token <EOS> gerado no passo N
```

---

## Conceitos Matemáticos Chave

### 1. Scaled Dot-Product Attention
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Onde:
- `Q` = Queries
- `K` = Keys
- `V` = Values
- `M` = Máscara (opcional, -∞ para posições futuras)
- $d_k$ = Dimensão de cada cabeça

### 2. Máscara Causal
```python
mask[i, j] = {
    0       if j <= i   (permitido)
    -∞      if j > i    (futuro, bloqueado)
}
```

### 3. Softmax Numericamente Estável
```python
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

### 4. Argmax para Seleção Greedy
```python
token_id = argmax(prob_next)  # Token com maior probabilidade
```

---

## Estrutura de Tensores

### Self-Attention (Decoder)
```
Input: [batch, seq_decoder, d_model]
Q, K, V: [batch, seq_decoder, d_k]
scores: [batch, seq_decoder, seq_decoder]
attention: [batch, seq_decoder, d_k]
Output: [batch, seq_decoder, d_model]
```

### Cross-Attention
```
Q: [batch, seq_decoder, d_k]     (do decoder)
K: [batch, seq_encoder, d_k]     (do encoder)
V: [batch, seq_encoder, d_v]     (do encoder)
scores: [batch, seq_decoder, seq_encoder]
Output: [batch, seq_decoder, d_v]
```

---

## Funcionalidades Principais

| Componente | Arquivo | Status |
|---|---|---|
| Máscara Causal | tarefa1_mascara_causal.py | Implementado |
| Softmax Estável | tarefa1_mascara_causal.py | Implementado |
| Cross-Attention | tarefa2_cross_attention.py | Implementado |
| MockVocabulary | tarefa3_loop_autoregressive.py | Implementado |
| Generate Next Token | tarefa3_loop_autoregressive.py | Implementado |
| Loop Auto-Regressivo | tarefa3_loop_autoregressive.py | Implementado |
| Argmax Decoding | tarefa3_loop_autoregressive.py | Implementado |
| Coordenador | main.py | Implementado |

---

## Aprendizados Esperados

Após completar este laboratório, você será capaz de:

1. Implementar mascaramento causal em atenção
2. Construir mecanismos de cross-attention
3. Simular loops de decodificação auto-regressivas
4. Entender o pipeline completo do Decoder Transformer
5. Trabalhar com álgebra linear em redes neurais

---

## Notas Importantes

- **Reprodutibilidade**: Todos os arquivos usam `np.random.seed(42)`
- **Sem Dependências Externas**: Apenas NumPy e Math (bibliotecas padrão)
- **Simulação**: Os tensores de encoder/decoder e pesos são fictícios (aleatórios)
- **Educacional**: Foco em clareza, não em performance otimizada

---

## Referências

- Vaswani et al. (2017) - "Attention Is All You Need" (Paper original Transformer)
- Lecture notes e slides do curso

---

**Autor:** Laboratório de Processamento de Linguagem Natural  
**Data:** 2026  
**Versão:** 1.0
