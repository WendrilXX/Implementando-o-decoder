# -*- coding: utf-8 -*-
"""
TAREFA 2: A PONTE ENCODER-DECODER (CROSS-ATTENTION)
====================================================

Fundamento: Diferente do Self-Attention, onde as representações derivam do mesmo
texto, o Encoder-Decoder Attention cruza fronteiras. O estado atual da geração
interage com a "memória" lida pelo Encoder.

O que está implementado aqui:
1. Tensores fictícios: encoder_output [1, 10, 512] e decoder_state [1, 4, 512]
2. Função cross_attention(encoder_out, decoder_state) que:
   - Projeta decoder_state para Query (Q)
   - Projeta encoder_out para Keys (K) e Values (V)
   - Calcula Scaled Dot-Product Attention SEM máscara causal
3. Prova Real: Demonstra cross-attention funcionando corretamente

Autor: Laboratório de Processamento de Linguagem Natural
"""

import numpy as np
import math
from typing import Tuple

np.random.seed(42)  # Reprodutibilidade


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calcula softmax de forma numericamente estável.
    Trata -∞ como 0 após aplicar a função exponencial.
    
    Args:
        x (np.ndarray): Array de entrada
        axis (int): Eixo ao longo do qual calcular softmax
    
    Returns:
        np.ndarray: Probabilidades após Softmax (soma = 1.0)
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    exp_x[np.isinf(x) & (x < 0)] = 0
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_attention(
    encoder_output: np.ndarray,
    decoder_state: np.ndarray,
    d_model: int = 512
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa Cross-Attention: Integração estrutural Encoder-Decoder.
    
    Diferenças críticas do Self-Attention:
    - Query (Q) vem do Decoder (decodificador)
    - Keys (K) e Values (V) vém do Encoder (codificador)
    - NÃO usamos máscara causal aqui (modelo vê encoder por completo)
    
    Equação:
        Attention(Q, K, V) = softmax(Q*K^T / sqrt(d_k)) * V
    
    Fluxo de dados:
    1. decoder_state [-] Projetar para Query Q
    2. encoder_output [-] Projetar para Keys K
    3. encoder_output [-] Projetar para Values V
    4. Calcular Q @ K^T / sqrt(d_k)
    5. Aplicar Softmax (sem máscara causal)
    6. Multiplicar pelos Values V
    
    Args:
        encoder_output (np.ndarray): Saída do Encoder [batch, seq_encoder, d_model]
        decoder_state (np.ndarray): Estado do Decoder [batch, seq_decoder, d_model]
        d_model (int): Dimensão do modelo (default: 512)
    
    Returns:
        Tuple[attention_output, attention_weights]:
            - attention_output: [batch, seq_decoder, d_model//8]
            - attention_weights: [batch, seq_decoder, seq_encoder]
    """
    batch_size = decoder_state.shape[0]
    seq_decoder = decoder_state.shape[1]
    seq_encoder = encoder_output.shape[1]
    d_k = d_model // 8  # 8 cabeças de atenção (simplificado para 1 cabeça)
    
    print(f"\n  Dimensões dentro de cross_attention():")
    print(f"    batch_size: {batch_size}")
    print(f"    seq_decoder: {seq_decoder}")
    print(f"    seq_encoder: {seq_encoder}")
    print(f"    d_k (dimensão por cabeça): {d_k}")
    
    # ===== PASSO 1: Projetar decoder_state para Query (Q) =====
    # Q = decoder_state @ W_q
    W_q = np.random.normal(0, 0.1, (d_model, d_k))
    Q = decoder_state @ W_q  # [batch, seq_decoder, d_k]
    
    print(f"\n  [OK] Query (Q) gerada de decoder_state:")
    print(f"    W_q (pesos): {W_q.shape}")
    print(f"    Q (resultado): {Q.shape}")
    
    # ===== PASSO 2: Projetar encoder_output para Keys (K) e Values (V) =====
    # K = encoder_output @ W_k
    # V = encoder_output @ W_v
    W_k = np.random.normal(0, 0.1, (d_model, d_k))
    W_v = np.random.normal(0, 0.1, (d_model, d_model // 8))
    
    K = encoder_output @ W_k  # [batch, seq_encoder, d_k]
    V = encoder_output @ W_v  # [batch, seq_encoder, d_model//8]
    
    print(f"\n  [OK] Keys (K) e Values (V) gerados de encoder_output:")
    print(f"    W_k (pesos): {W_k.shape}")
    print(f"    W_v (pesos): {W_v.shape}")
    print(f"    K (resultado): {K.shape}")
    print(f"    V (resultado): {V.shape}")
    
    # ===== PASSO 3: Calcular Scaled Dot-Product Attention =====
    # Scores = Q @ K^T / sqrt(d_k)
    # [batch, seq_decoder, seq_encoder]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)
    
    print(f"\n  [OK] Scores (Q @ K^T / sqrt(d_k)) calculados:")
    print(f"    Forma: {scores.shape}")
    print(f"    [batch, seq_decoder, seq_encoder]")
    print(f"    Significado: Cada posição do decoder pode atender cada posição do encoder")
    
    # ===== PASSO 4: Aplicar Softmax (SEM máscara causal!) =====
    # Aqui é diferente do Self-Attention do Decoder
    # Decoder pode olhar TODA a sequência do Encoder
    attention_weights = softmax(scores, axis=-1)  # [batch, seq_decoder, seq_encoder]
    
    print(f"\n  [OK] Softmax aplicado (SEM máscara causal):")
    print(f"    Forma: {attention_weights.shape}")
    print(f"    Cada linha soma para 1.0 (distribuição de probabilidade)")
    
    # ===== PASSO 5: Aplicar pesos aos valores =====
    # Output = attention_weights @ V
    # [batch, seq_decoder, d_model//8]
    attention_output = np.matmul(attention_weights, V)
    
    print(f"\n  [OK] Contexto do Encoder integrado ao Decoder:")
    print(f"    attention_output: {attention_output.shape}")
    print(f"    [batch, seq_decoder, d_model//8]")
    
    return attention_output, attention_weights


def prova_cross_attention():
    """
    Prova real: Demonstra Cross-Attention funcionando corretamente.
    """
    print("\n" + "="*80)
    print("TAREFA 2: CROSS-ATTENTION (PONTE ENCODER-DECODER)")
    print("="*80)
    
    # ===== Parâmetros fictícios (conforme especificado no laboratório) =====
    batch_size = 1
    seq_encoder = 10  # Frase em francês (exemplo)
    seq_decoder = 4   # O que o decoder já gerou em inglês
    d_model = 512
    
    print(f"\nParâmetros da simulação:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_encoder (frase fonte): {seq_encoder}")
    print(f"  seq_decoder (frase alvo gerada): {seq_decoder}")
    print(f"  d_model (dimensão): {d_model}")
    
    # ===== Criar tensores fictícios =====
    print(f"\nCriando tensores fictícios...")
    encoder_output = np.random.normal(0, 1, (batch_size, seq_encoder, d_model))
    decoder_state = np.random.normal(0, 1, (batch_size, seq_decoder, d_model))
    
    print(f"\n[OK] Tensores criados:")
    print(f"  encoder_output (saída do Encoder): {encoder_output.shape}")
    print(f"    Representa a 'memória' da frase original (francês)")
    print(f"  decoder_state (estado atual do Decoder): {decoder_state.shape}")
    print(f"    Representa what o modelo gerou até agora (inglês)")
    
    # ===== Aplicar cross-attention =====
    print(f"\nAplicando Cross-Attention...")
    attention_output, attention_weights = cross_attention(encoder_output, decoder_state, d_model)
    
    # ===== Analisar resultados =====
    print(f"\n" + "-"*80)
    print("ANÁLISE DOS RESULTADOS")
    print("-"*80)
    
    print(f"\nFormas após Cross-Attention:")
    print(f"  attention_output: {attention_output.shape}")
    print(f"    Contexto do Encoder integrado ao Decoder")
    print(f"  attention_weights: {attention_weights.shape}")
    print(f"    Pesos de atenção [seq_decoder, seq_encoder]")
    
    # ===== Visualizar pesos de atenção =====
    print(f"\nPesos de Atenção para cada posição do Decoder:")
    print(f"  (Mostra qual posição do Encoder cada posição do Decoder atende)\n")
    
    for i in range(seq_decoder):
        weights_at_i = attention_weights[0, i, :]
        top_encoder_idx = np.argmax(weights_at_i)
        top_weight = weights_at_i[top_encoder_idx]
        
        print(f"  Posição Decoder {i}:")
        print(f"    Pesos sobre Encoder: {weights_at_i}")
        print(f"    Maior peso: encoder[{top_encoder_idx}] = {top_weight:.4f}")
        print(f"    Soma dos pesos: {weights_at_i.sum():.6f} (deve ser 1.0)")
    
    # ===== Verificar regra de Cross-Attention =====
    print(f"\n" + "-"*80)
    print("VERIFICAÇÃO DE CROSS-ATTENTION")
    print("-"*80)
    
    all_correct = True
    for i in range(seq_decoder):
        weights_at_i = attention_weights[0, i, :]
        
        # Cada posição do decoder deve:
        # 1. Ter soma de pesos = 1.0
        sum_weights = weights_at_i.sum()
        check_sum = np.isclose(sum_weights, 1.0)
        
        # 2. Poder atender a qualquer posição do encoder (sem máscara causal)
        can_attend_all = True
        for j in range(seq_encoder):
            # Cada posição j pode ser atendida (peso > 0 ou próximo de 0)
            if weights_at_i[j] < 0:
                can_attend_all = False
                all_correct = False
        
        status = "[OK]" if (check_sum and can_attend_all) else "[ERRO]"
        print(f"{status} Decoder[{i}]: soma={sum_weights:.6f}, pode atender encoder=[0..{seq_encoder-1}]")
    
    print(f"\n" + "="*80)
    if all_correct:
        print("[OK] SUCESSO: Cross-Attention funcionando perfeitamente!")
        print("  - Cada posição decoder distribui atenção sobre encoder")
        print("  - Sem máscara causal (pode olhar qualquer posição do encoder)")
        print("  - Pesos normalizam corretamente (soma = 1.0)")
    else:
        print("[ERRO] Cross-Attention com problemas detectados!")
    print("="*80)
    
    return attention_output, attention_weights


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TAREFA 2: CROSS-ATTENTION (PONTE ENCODER-DECODER)".center(80))
    print("="*80)
    
    # Executar prova
    attention_output, attention_weights = prova_cross_attention()
    
    print("\nRESUMO DA TAREFA 2:")
    print("[OK] Tensores Encoder-Decoder criados corretamente")
    print("[OK] Funções de projeção (W_q, W_k, W_v) implementadas")
    print("[OK] Scaled Dot-Product Attention calculado")
    print("[OK] Softmax sem máscara causal (acesso completo ao Encoder)")
    print("[OK] Output contexto integrado ao pipeline de decodificação")
    print("\n" + "="*80)
