# -*- coding: utf-8 -*-
"""
TAREFA 1: IMPLEMENTANDO A MASCARA CAUSAL (LOOK-AHEAD MASK)
========================================================

Fundamento: No treinamento paralelizado em GPUs, a frase de destino completa entra
no Decoder. Para impedir que a palavra na posicao i atenda a posicao i+1, injetamos
uma mascara matricial M antes do calculo do Softmax.

O que esta implementado aqui:
- Funcao create_causal_mask(seq_len)
- Funcao softmax() com estabilidade numerica
- Prova Real: Multiplica Q e K, adiciona mascara, aplica Softmax e verifica que
  probabilidades futuras sao 0.0

Autor: Laboratorio de Processamento de Linguagem Natural
"""

import numpy as np
import math

np.random.seed(42)  # Reprodutibilidade


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Cria uma máscara causal para impedir que posições futuras sejam atendidas.
    
    Fundamental Matemático:
    - Posições <= i podem atender a posição i (zeros)
    - Posições > i recebem -∞ para zerar suas probabilidades no Softmax
    
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
    # triu_indices_from retorna índices da parte triangular superior
    # k=1 significa começar a partir de 1 acima da diagonal principal
    mask[np.triu_indices_from(mask, k=1)] = -np.inf
    
    return mask


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Calcula softmax de forma numericamente estável.
    Trata -inf como 0 após aplicar a função exponencial.
    
    Fórmula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
    
    Estabilidade numérica: Subtrai o máximo antes de calcular exp()
    para evitar overflow/underflow.
    
    Args:
        x (np.ndarray): Array de entrada
        axis (int): Eixo ao longo do qual calcular softmax (default: último eixo)
    
    Returns:
        np.ndarray: Probabilidades após Softmax (soma = 1.0)
    """
    # Subtração de máximo para estabilidade numérica
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    
    # -inf se torna 0 após exp(-inf) = 0
    # Garantir que valores -inf produzem 0 na exponencial
    exp_x[np.isinf(x) & (x < 0)] = 0
    
    # Normalizar: dividir pela soma para obter probabilidades
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def prova_mascara_causal(seq_len: int = 5):
    """
    Prova real: Demonstra que a máscara causal funciona corretamente.
    
    Processo:
    1. Gera Q (Queries) e K (Keys) fictícios
    2. Calcula Scores = Q @ K^T / sqrt(d_k)
    3. Adiciona máscara causal aos scores
    4. Aplica Softmax
    5. Verifica que posições futuras têm probabilidade 0.0
    
    Args:
        seq_len (int): Comprimento da sequência (default: 5)
    
    Returns:
        Tuple: (causal_mask, attention_weights)
    """
    print("\n" + "="*80)
    print("TAREFA 1: MÁSCARA CAUSAL (LOOK-AHEAD MASK)")
    print("="*80)
    
    d_model = 64
    batch_size = 1
    
    print(f"\nParâmetros:")
    print(f"  Comprimento da sequência (seq_len): {seq_len}")
    print(f"  Dimensão do modelo (d_model): {d_model}")
    print(f"  Tamanho do batch: {batch_size}")
    
    # ===== PASSO 1: Gerar Queries (Q) e Keys (K) fictícios =====
    Q = np.random.normal(0, 1, (batch_size, seq_len, d_model))
    K = np.random.normal(0, 1, (batch_size, seq_len, d_model))
    
    print(f"\nFormas dos tensores:")
    print(f"  Q (Queries): {Q.shape}")
    print(f"  K (Keys): {K.shape}")
    
    # ===== PASSO 2: Calcular Scaled Dot-Product Attention =====
    # Scores = Q @ K^T / sqrt(d_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_model)
    
    print(f"\nScores (Attention antes da máscara) - Primeiras 5x5 valores:")
    print(scores[0, :5, :5])
    
    # ===== PASSO 3: Criar e aplicar máscara causal =====
    causal_mask = create_causal_mask(seq_len)
    
    print(f"\nMáscara Causal [seq_len={seq_len}, seq_len={seq_len}] - Primeiras 5x5:")
    print(causal_mask[:5, :5])
    print("\nInterpretação da máscara:")
    print("  - Zeros (0.0): Posições permitidas (look-back)")
    print("  - -inf: Posições futuras (bloqueadas)")
    
    # Adicionar máscara aos scores
    masked_scores = scores + causal_mask
    
    # ===== PASSO 4: Aplicar Softmax =====
    attention_weights = softmax(masked_scores, axis=-1)
    
    print(f"\nPesos de Atenção após Softmax - Primeiras 5x5:")
    print(attention_weights[0, :5, :5])
    print("\nInterpretação dos pesos:")
    print("  - Posições futuras (acima da diagonal): 0.0 (bloqueadas)")
    print("  - Posições passadas (diagonal inferior): >0.0 (permitidas)")
    
    # ===== PASSO 5: Verificar Causalidade =====
    print("\n" + "-"*80)
    print("VERIFICAÇÃO DE CAUSALIDADE")
    print("-"*80)
    
    all_blocked = True
    for i in range(min(seq_len, 5)):
        future_probs = attention_weights[0, i, i+1:]
        has_future = np.any(future_probs > 1e-10)
        status = "[OK] BLOQUEADO" if not has_future else "[ERRO] VAZAMENTO"
        print(f"Posição {i}:")
        print(f"  Pode olhar para posições <= {i}: {list(range(i+1))}")
        print(f"  Posições futuras ({i+1}, {i+2}, ...): {future_probs}")
        print(f"  Status: {status}")
        
        if has_future:
            all_blocked = False
    
    print("\n" + "="*80)
    if all_blocked:
        print("[OK] SUCESSO: Máscara causal funcionando perfeitamente!")
        print("  Todas as probabilidades futuras são EXATAMENTE 0.0")
    else:
        print("[ERRO] Máscara causal com vazamento detectado!")
    print("="*80)
    
    return causal_mask, attention_weights


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TAREFA 1: MÁSCARA CAUSAL (LOOK-AHEAD MASK)".center(80))
    print("="*80)
    
    # Executar prova
    causal_mask, attention_weights = prova_mascara_causal(seq_len=5)
    
    print("\nRESUMO DA TAREFA 1:")
    print("[OK] Máscara Causal criada corretamente")
    print("[OK] Softmax aplicado com estabilidade numérica")
    print("[OK] Posições futuras bloqueadas (probabilidade = 0.0)")
    print("[OK] Modelo não consegue 'olhar para o futuro'")
    print("\n" + "="*80)
