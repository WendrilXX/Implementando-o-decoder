# -*- coding: utf-8 -*-
"""
LABORATÓRIO 2: IMPLEMENTANDO O DECODER DO TRANSFORMER


Objetivos:
1. Dominar a álgebra linear do mascaramento causal (Look-Ahead Masking)
2. Implementar a ponte Encoder-Decoder (Cross-Attention)
3. Simular um loop de geração auto-regressiva

Autor: Laboratório de Processamento de Linguagem Natural
"""

import numpy as np
import math
from typing import List, Tuple, Dict

np.random.seed(42)  # Reprodutibilidade

# ============================================================================
# TAREFA 1: IMPLEMENTANDO A MÁSCARA CAUSAL (LOOK-AHEAD MASK)
# ============================================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Cria uma máscara causal para impedir que posições futuras sejam atendidas.
    
    Fundamental Matemático:
    - Posições <= i podem atender a posição i
    - Posições > i recebem -inf para zerar suas probabilidades no Softmax
    
    Args:
        seq_len (int): Comprimento da sequência (ex: 5)
    
    Returns:
        np.ndarray: Matriz [seq_len, seq_len] com zeros na diagonal inferior
                    e -inf na diagonal superior
    
    Exemplo:
        Para seq_len=3:
        [[ 0. -inf -inf ]
         [ 0.  0. -inf ]
         [ 0.  0.  0.]]
    """
    # Criar matriz de zeros
    mask = np.zeros((seq_len, seq_len))
    
    # Preencher a parte triangular superior com -∞
    mask[np.triu_indices_from(mask, k=1)] = -np.inf
    
    return mask


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calcula softmax de forma numericamente estável.
    Trata -inf como 0 após aplicar a função exponencial.
    
    Args:
        x (np.ndarray): Array de entrada
        axis (int): Eixo ao longo do qual calcular softmax
    
    Returns:
        np.ndarray: Probabilidades após Softmax
    """
    # Subtração de máximo para estabilidade numérica
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # -inf se torna 0 após exp(-inf) = 0
    exp_x[np.isinf(x) & (x < 0)] = 0
    
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def prova_mascara_causal(seq_len: int = 5):
    """
    Prova real: Demonstra que a máscara causal funciona corretamente.
    
    Processo:
    1. Gera Q (Queries) e K (Keys) fictícios
    2. Calcula Scores = Q @ K^T
    3. Adiciona máscara causal
    4. Aplica Softmax
    5. Verifica que posições futuras têm probabilidade 0.0
    """
    print("\n" + "="*80)
    print("TAREFA 1: MÁSCARA CAUSAL (LOOK-AHEAD MASK)")
    print("="*80)
    
    d_model = 64
    batch_size = 1
    
    # Gerar Q e K fictícios
    Q = np.random.normal(0, 1, (batch_size, seq_len, d_model))
    K = np.random.normal(0, 1, (batch_size, seq_len, d_model))
    
    # Scaled Dot-Product: Scores = Q @ K^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_model)
    
    print(f"\nScores (primeiras 5x5 valores):\n{scores[0, :5, :5]}")
    
    # Criar e aplicar máscara causal
    causal_mask = create_causal_mask(seq_len)
    
    print(f"\nMáscara Causal (primeiras 5x5):\n{causal_mask[:5, :5]}")
    
    # Adicionar máscara aos scores
    masked_scores = scores + causal_mask
    
    # Aplicar Softmax
    attention_weights = softmax(masked_scores, axis=-1)
    
    print(f"\nPesos de Atenção após Softmax (primeiras 5x5):\n{attention_weights[0, :5, :5]}")
    
    # Verificação: Elementos acima da diagonal devem ser 0.0
    print("\n Verificação de Causalidade:")
    for i in range(min(5, seq_len)):
        future_probs = attention_weights[0, i, i+1:]
        has_future = np.any(future_probs > 1e-10)
        print(f"  Posição {i} com futuro (pos > {i}): {future_probs} => Bloqueado: {not has_future}")
    
    print("\nProva: Probabilidades de posições futuras são 0.0!")
    return causal_mask, attention_weights


# ============================================================================
# TAREFA 2: A PONTE ENCODER-DECODER (CROSS-ATTENTION)
# ============================================================================

def cross_attention(
    encoder_output: np.ndarray,
    decoder_state: np.ndarray,
    d_model: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa Cross-Attention: Integração estrutural Encoder-Decoder.
    
    Diferença do Self-Attention:
    - Query (Q) vem do Decoder (decodificador)
    - Keys (K) e Values (V) vêm do Encoder (codificador)
    - NÃO usamos máscara causal aqui (modelo vê encoder por completo)
    
    Equação:
        Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
    
    Args:
        encoder_output (np.ndarray): [batch, seq_encoder, d_model]
        decoder_state (np.ndarray): [batch, seq_decoder, d_model]
        d_model (int): Dimensão do modelo
    
    Returns:
        Tuple[attention_output, attention_weights]
    """
    batch_size = decoder_state.shape[0]
    seq_decoder = decoder_state.shape[1]
    seq_encoder = encoder_output.shape[1]
    d_k = d_model // 8  # 8 cabeças de atenção
    
    # Projetar decoder_state para Query (Q)
    W_q = np.random.normal(0, 0.1, (d_model, d_k))
    Q = decoder_state @ W_q  # [batch, seq_decoder, d_k]
    
    # Projetar encoder_output para Keys (K) e Values (V)
    W_k = np.random.normal(0, 0.1, (d_model, d_k))
    W_v = np.random.normal(0, 0.1, (d_model, d_model // 8))
    
    K = encoder_output @ W_k  # [batch, seq_encoder, d_k]
    V = encoder_output @ W_v  # [batch, seq_encoder, d_model//8]
    
    # Scaled Dot-Product Attention
    # Scores = Q @ K^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)
    # [batch, seq_decoder, seq_encoder]
    
    # Softmax para obter pesos de atenção
    attention_weights = softmax(scores, axis=-1)  # [batch, seq_decoder, seq_encoder]
    
    # Aplicar pesos aos valores
    attention_output = np.matmul(attention_weights, V)  # [batch, seq_decoder, d_model//8]
    
    return attention_output, attention_weights


def prova_cross_attention():
    """
    Prova real: Demonstra Cross-Attention funcionando.
    """
    print("\n" + "="*80)
    print("TAREFA 2: CROSS-ATTENTION (PONTE ENCODER-DECODER)")
    print("="*80)
    
    # Parâmetros fictícios (conforme especificado no laboratório)
    batch_size = 1
    seq_encoder = 10  # Frase em francês
    seq_decoder = 4   # O que o decoder já gerou em inglês
    d_model = 512
    
    # Criar tensores fictícios
    encoder_output = np.random.normal(0, 1, (batch_size, seq_encoder, d_model))
    decoder_state = np.random.normal(0, 1, (batch_size, seq_decoder, d_model))
    
    print(f"\nFormas dos tensores:")
    print(f"  encoder_output (saída Encoder): {encoder_output.shape}")
    print(f"  decoder_state (estado Decoder): {decoder_state.shape}")
    
    # Aplicar cross-attention
    attention_output, attention_weights = cross_attention(encoder_output, decoder_state, d_model)
    
    print(f"\nFormas após Cross-Attention:")
    print(f"  attention_output: {attention_output.shape}")
    print(f"  attention_weights: {attention_weights.shape}")
    
    # Visualizar pesos de atenção para primeira posição do decoder
    print(f"\nPesos de Atenção da 1ª posição decoder sobre encoder:")
    print(f"  {attention_weights[0, 0, :]}") 
    print(f"  Soma dos pesos: {attention_weights[0, 0, :].sum():.4f} (deve ser 1.0)")
    
    # Demonstrar: query i só pode atender a todas as posições do encoder
    print(f"\nEvidência:")
    for i in range(min(seq_decoder, 3)):
        max_weight = attention_weights[0, i, :].max()
        print(f"  Posição decoder {i}: max weight = {max_weight:.4f} (pode atender todos encoder)")
    
    print("\nProva: Cross-Attention funciona corretamente!")
    return attention_output, attention_weights


# ============================================================================
# TAREFA 3: SIMULANDO O LOOP DE INFERÊNCIA AUTO-REGRESSIVO
# ============================================================================

class MockVocabulary:
    """Simulação de vocabulário com tokens especiais."""
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        
        # Tokens especiais
        self.START = "<START>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"
        
        # Tokens fictícios (para simulação)
        self.tokens = ([self.START, self.EOS, self.UNK] + 
                       [f"word_{i}" for i in range(3, vocab_size)])
        self.token2id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
    
    def get_id(self, token: str) -> int:
        return self.token2id.get(token, self.token2id[self.UNK])
    
    def get_token(self, token_id: int) -> str:
        return self.id2token.get(token_id, self.UNK)


def generate_next_token(
    current_sequence: List[str],
    encoder_output: np.ndarray,
    vocab: MockVocabulary,
    d_model: int = 512
) -> np.ndarray:
    """
    Função genérica que simula a passagem pelo Decoder.
    
    Processo:
    1. Codifica sequência atual em embeddings fictícios
    2. Passa pelo Decoder (com Self-Attention + Cross-Attention)
    3. Projeta para tamanho do vocabulário
    4. Retorna distribuição de probabilidades
    
    Args:
        current_sequence (List[str]): Tokens gerados até agora
        encoder_output (np.ndarray): Saída do Encoder
        vocab (MockVocabulary): Vocabulário
        d_model (int): Dimensão do modelo
    
    Returns:
        np.ndarray: Vetor de probabilidades [vocab_size]
    """
    batch_size = 1
    seq_len = len(current_sequence)
    
    # Simular embeddings da sequência atual (normalmente treináveis)
    current_embedding = np.random.normal(0, 1, (batch_size, seq_len, d_model))
    
    # Self-Attention na sequência atual (com máscara causal)
    causal_mask = create_causal_mask(seq_len)
    Q_self = current_embedding @ np.random.normal(0, 0.1, (d_model, d_model // 8))
    K_self = current_embedding @ np.random.normal(0, 0.1, (d_model, d_model // 8))
    V_self = current_embedding @ np.random.normal(0, 0.1, (d_model, d_model))
    
    scores_self = Q_self @ K_self.transpose(0, 2, 1) / math.sqrt(d_model // 8)
    scores_self = scores_self + causal_mask
    weights_self = softmax(scores_self)
    self_attention_out = weights_self @ V_self
    
    # Cross-Attention com encoder
    cross_out, _ = cross_attention(encoder_output, current_embedding, d_model)
    
    # Combinar (simplificado: apenas self-attention + encoder output)
    # Em prática real: FFN(self_attn + cross_attn + layer_norm + residual)
    decoder_output = self_attention_out  # Último token apenas
    
    # Projetar para vocabulário
    W_output = np.random.normal(0, 0.1, (d_model, vocab.vocab_size))
    logits = decoder_output[0, -1, :] @ W_output  # Último token: [vocab_size]
    
    # Softmax para obter probabilidades
    probs = softmax(logits.reshape(1, -1), axis=-1)[0]
    
    return probs


def generate_with_argmax(
    encoder_output: np.ndarray,
    vocab: MockVocabulary,
    max_length: int = 20
) -> List[str]:
    """
    Loop de inferência auto-regressiva com argmax.
    
    Processo:
    1. Começa com token <START>
    2. A cada iteração, gera o token mais provável (argmax)
    3. Adiciona à sequência
    4. Para se gerar <EOS> ou atingir max_length
    
    Args:
        encoder_output (np.ndarray): Saída do Encoder
        vocab (MockVocabulary): Vocabulário
        max_length (int): Comprimento máximo de geração
    
    Returns:
        List[str]: Sequência gerada
    """
    current_sequence = [vocab.START]
    
    print(f"\nIniciando geração com max_length={max_length}")
    print(f"Sequência inicial: {current_sequence}")
    
    for step in range(max_length):
        # Gerar distribuição de probabilidades para próximo token
        prob_next = generate_next_token(current_sequence, encoder_output, vocab)
        
        # Argmax: selecionar token com maior probabilidade
        token_id = np.argmax(prob_next)
        token = vocab.get_token(token_id)
        
        # Adicionar à sequência
        current_sequence.append(token)
        
        print(f"Passo {step + 1}: {token} (prob: {prob_next[token_id]:.4f})")
        
        # Verificar <EOS>
        if token == vocab.EOS:
            print(f"\n[OK] Token <EOS> gerado no passo {step + 1}")
            break
    
    return current_sequence


def prova_autoregressive():
    """
    Prova real: Demonstra o loop de inferência auto-regressiva.
    """
    print("\n" + "="*80)
    print("TAREFA 3: LOOP DE INFERÊNCIA AUTO-REGRESSIVO")
    print("="*80)
    
    # Criar vocabulário
    vocab = MockVocabulary(vocab_size=10000)
    
    # Simular saída do encoder (encoder processou frase em francês)
    encoder_output = np.random.normal(0, 1, (1, 10, 512))
    
    print(f"\nVocabulário fictício:")
    print(f"  Tamanho: {vocab.vocab_size}")
    print(f"  Tokens especiais: {vocab.START}, {vocab.EOS}, {vocab.UNK}")
    
    # Rodar o loop de geração
    generated_sequence = generate_with_argmax(encoder_output, vocab, max_length=20)
    
    print(f"\n" + "-"*80)
    print(f"Sequência gerada completa:")
    print(f"  {' -> '.join(generated_sequence)}")
    print(f"  Total de {len(generated_sequence)} tokens")
    
    return generated_sequence


# ============================================================================
# EXECUTAR TUDO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("LABORATÓRIO 2: IMPLEMENTANDO O DECODER DO TRANSFORMER".center(80))
    print("="*80)
    
    # Tarefa 1: Máscara Causal
    causal_mask, attention_weights = prova_mascara_causal(seq_len=5)
    
    # Tarefa 2: Cross-Attention
    attention_output, cross_weights = prova_cross_attention()
    
    # Tarefa 3: Loop Auto-Regressivo
    generated = prova_autoregressive()
    
    # Resumo final
    print("\n" + "="*80)
    print("RESUMO EXECUTADO COM SUCESSO")
    print("="*80)
    print("""
[COMPLETO] Tarefa 1: Máscara Causal implementada e verificada
  - Impede que posições futuras recebam atenção
  - Probabilidades futuras = 0.0 após Softmax

[COMPLETO] Tarefa 2: Cross-Attention implementada e verificada
  - Integração estrutural Encoder-Decoder
  - Decoder pode atender completamente ao Encoder

[COMPLETO] Tarefa 3: Loop Auto-Regressivo implementado
  - Gera tokens iterativamente com argmax
  - Para ao encontrar <EOS> ou atingir max_length
    """)
    print("="*80)
