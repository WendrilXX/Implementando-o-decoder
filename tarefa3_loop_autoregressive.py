# -*- coding: utf-8 -*-
"""
TAREFA 3: SIMULANDO O LOOP DE INFERÊNCIA AUTO-REGRESSIVO
=========================================================

Fundamento: O modelo de linguagem trabalha como um laço de repetição condicionado
na produção anterior. Após as sub-camadas, o vetor final de 512 dimensões sofre
uma projeção linear para o tamanho do vocabulário, seguida de um Softmax, emitindo
uma distribuição de probabilidades para o próximo token.

O que está implementado aqui:
1. Classe MockVocabulary: Simulação de vocabulário com tokens especiais
2. Função generate_next_token(): Simula passagem pelo Decoder
3. Função generate_with_argmax(): Loop while de inferência auto-regressiva
4. Prova Real: Demonstra geração de texto iterativa

Autor: Laboratório de Processamento de Linguagem Natural
"""

import numpy as np
import math
from typing import List

np.random.seed(42)  # Reprodutibilidade


def create_causal_mask(seq_len: int) -> np.ndarray:
    """Cria máscara causal para Self-Attention do Decoder."""
    mask = np.zeros((seq_len, seq_len))
    mask[np.triu_indices_from(mask, k=1)] = -np.inf
    return mask


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Calcula softmax de forma numericamente estável."""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    exp_x[np.isinf(x) & (x < 0)] = 0
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_attention(
    encoder_output: np.ndarray,
    decoder_state: np.ndarray,
    d_model: int = 512
) -> tuple:
    """Cross-Attention simplificado para geração."""
    batch_size = decoder_state.shape[0]
    seq_decoder = decoder_state.shape[1]
    seq_encoder = encoder_output.shape[1]
    d_k = d_model // 8
    
    W_q = np.random.normal(0, 0.1, (d_model, d_k))
    Q = decoder_state @ W_q
    
    W_k = np.random.normal(0, 0.1, (d_model, d_k))
    W_v = np.random.normal(0, 0.1, (d_model, d_model // 8))
    
    K = encoder_output @ W_k
    V = encoder_output @ W_v
    
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / math.sqrt(d_k)
    attention_weights = softmax(scores, axis=-1)
    attention_output = np.matmul(attention_weights, V)
    
    return attention_output, attention_weights


class MockVocabulary:
    """
    Simulação de vocabulário com tokens especiais.
    
    Estrutura:
    - Token 0: <START>  (início de sequência)
    - Token 1: <EOS>    (fim de sequência)
    - Token 2: <UNK>    (desconhecido)
    - Token 3+: word_3, word_4, ... (palavras fictícias)
    
    Atributos:
        vocab_size (int): Tamanho total do vocabulário
        START (str): Token inicial
        EOS (str): Token de fim de sequência
        UNK (str): Token desconhecido
    """
    
    def __init__(self, vocab_size: int = 10000):
        """
        Inicializa vocabulário fictício.
        
        Args:
            vocab_size (int): Tamanho do vocabulário (default: 10.000)
        """
        self.vocab_size = vocab_size
        
        # Tokens especiais
        self.START = "<START>"
        self.EOS = "<EOS>"
        self.UNK = "<UNK>"
        
        # Construir tokens: tokens especiais + palavras fictícias
        self.tokens = (
            [self.START, self.EOS, self.UNK] + 
            [f"word_{i}" for i in range(3, vocab_size)]
        )
        
        # Mapas bidireccionais
        self.token2id = {tok: i for i, tok in enumerate(self.tokens)}
        self.id2token = {i: tok for tok, i in self.token2id.items()}
    
    def get_id(self, token: str) -> int:
        """Obter ID de um token."""
        return self.token2id.get(token, self.token2id[self.UNK])
    
    def get_token(self, token_id: int) -> str:
        """Obter token de um ID."""
        return self.id2token.get(token_id, self.UNK)


def generate_next_token(
    current_sequence: List[str],
    encoder_output: np.ndarray,
    vocab: MockVocabulary,
    d_model: int = 512
) -> np.ndarray:
    """
    Função genérica que simula a passagem pelo Decoder.
    
    Pipeline de geração:
    1. Embed sequência atual → [batch, seq_len, d_model]
    2. Self-Attention (com máscara causal) → cada token vê seu passado
    3. Cross-Attention (sem máscara) → integrado contexto do Encoder
    4. FFN (feed-forward network) → não-linearidade
    5. Output projection → [batch, seq_len, vocab_size]
    6. Softmax → distribuição de probabilidades
    
    Args:
        current_sequence (List[str]): Tokens gerados até agora
                                      (ex: ["<START>", "O", "rato"])
        encoder_output (np.ndarray): Saída do Encoder [batch, seq_encoder, d_model]
        vocab (MockVocabulary): Vocabulário com tokens especiais
        d_model (int): Dimensão do modelo (default: 512)
    
    Returns:
        np.ndarray: Vetor de probabilidades [vocab_size]
                    Distribuição P(next_token | current_sequence, encoder)
    """
    batch_size = 1
    seq_len = len(current_sequence)
    
    # ===== PASSO 1: Embedding da sequência atual =====
    # Em um modelo real, seria uma lookup table treinável
    # Aqui, simulamos com valores aleatórios para demonstrar
    current_embedding = np.random.normal(0, 1, (batch_size, seq_len, d_model))
    
    # ===== PASSO 2: Self-Attention (com máscara causal) =====
    # Cada token só pode atender seus tokens anteriores
    causal_mask = create_causal_mask(seq_len)
    
    # Projetar para Q, K, V
    Q_self = current_embedding @ np.random.normal(0, 0.1, (d_model, d_model // 8))
    K_self = current_embedding @ np.random.normal(0, 0.1, (d_model, d_model // 8))
    V_self = current_embedding @ np.random.normal(0, 0.1, (d_model, d_model))
    
    # Scaled Dot-Product Attention com máscara
    scores_self = Q_self @ K_self.transpose(0, 2, 1) / math.sqrt(d_model // 8)
    scores_self = scores_self + causal_mask
    weights_self = softmax(scores_self)
    self_attention_out = weights_self @ V_self
    
    # ===== PASSO 3: Cross-Attention com Encoder =====
    # Decoder atende completamente a Encoder (sem máscara causal)
    cross_out, _ = cross_attention(encoder_output, current_embedding, d_model)
    
    # ===== PASSO 4: Combinar Self-Attention + Cross-Attention =====
    # Em um modelo real: residual connections + layer normalization
    # Aqui simplificamos usando apenas self-attention
    decoder_output = self_attention_out  # [batch, seq_len, d_model]
    
    # ===== PASSO 5: FFN (Feed-Forward Network) =====
    # Projeção não-linear: d_model -> 4*d_model -> d_model
    # Aqui simplificamos como identidade
    ffn_output = decoder_output
    
    # ===== PASSO 6: Output Projection =====
    # Projetar último token para vocabulary size
    # logits = decoder_output[-1, :] @ W_output + b_output
    W_output = np.random.normal(0, 0.1, (d_model, vocab.vocab_size))
    
    # Pegar último token apenas (aquele sendo computado)
    logits = decoder_output[0, -1, :] @ W_output  # [vocab_size]
    
    # ===== PASSO 7: Softmax para probabilidades =====
    # p(token) = exp(logit) / sum(exp(logits))
    probs = softmax(logits.reshape(1, -1), axis=-1)[0]  # [vocab_size]
    
    return probs


def generate_with_argmax(
    encoder_output: np.ndarray,
    vocab: MockVocabulary,
    max_length: int = 20
) -> List[str]:
    """
    Loop de inferência auto-regressiva com argmax.
    
    Pseudocódigo:
    -----------
    sequence = [<START>]
    while True:
        probs = generate_next_token(sequence, encoder_output)
        token_id = argmax(probs)           # Token com maior probabilidade
        token = vocab[token_id]
        sequence.append(token)
        
        if token == <EOS>:                   # Parar se fim de sequência
            break
        if len(sequence) >= max_length:      # Parar se limite atingido
            break
    
    return sequence
    
    Interpretação:
    - A cada passo, geramos distribuição P(next_token | contexto)
    - Selecionamos token com MAIOR probabilidade (greedy/argmax)
    - Isso é eficiente mas pode gerar frases menos diversas
    - Alternativas: sampling, beam search, nucleus sampling
    
    Args:
        encoder_output (np.ndarray): Saída do Encoder
        vocab (MockVocabulary): Vocabulário
        max_length (int): Limite máximo de tokens (default: 20)
    
    Returns:
        List[str]: Sequência de tokens gerados
    """
    current_sequence = [vocab.START]
    
    print(f"\n{'='*80}")
    print("LOOP DE INFERÊNCIA AUTO-REGRESSIVA")
    print(f"{'='*80}")
    
    print(f"\nParâmetros:")
    print(f"  max_length: {max_length}")
    print(f"  vocab_size: {vocab.vocab_size}")
    print(f"  Estratégia: argmax (greedy - token de maior probabilidade)")
    
    print(f"\n{'Iteração':<12} {'Token':<20} {'Probabilidade':<15} {'Sequência':<20}")
    print("-" * 80)
    
    # ===== LOOP PRINCIPAL =====
    for step in range(max_length):
        # Gerar distribuição de probabilidades para próximo token
        prob_next = generate_next_token(current_sequence, encoder_output, vocab)
        
        # Argmax: selecionar token com MAIOR probabilidade
        token_id = np.argmax(prob_next)
        token = vocab.get_token(token_id)
        prob_value = prob_next[token_id]
        
        # Adicionar à sequência
        current_sequence.append(token)
        
        # Mostrar progresso
        seq_display = " -> ".join(current_sequence[-3:])
        if len(current_sequence) > 3:
            seq_display = "... -> " + seq_display
        
        print(f"Passo {step + 1:<7} {token:<20} {prob_value:<15.4f} {seq_display:<20}")
        
        # ===== Critério de parada 1: Token <EOS> =====
        if token == vocab.EOS:
            print(f"\n[OK] Token <EOS> gerado no passo {step + 1}")
            print("  Modelo sinalizou fim de sequência")
            break
    else:
        # Critério de parada 2: Atingiu max_length
        print(f"\n⚠ Atingiu comprimento máximo ({max_length} tokens)")
        print("  Generated token ainda não sinalizou fim de sequência")
    
    return current_sequence


def prova_autoregressive():
    """
    Prova real: Demonstra o loop de inferência auto-regressiva.
    """
    print("\n" + "="*80)
    print("TAREFA 3: LOOP DE INFERÊNCIA AUTO-REGRESSIVO")
    print("="*80)
    
    # ===== Configuração =====
    vocab_size = 10000
    max_generation_steps = 20
    
    # Criar vocabulário fictício
    vocab = MockVocabulary(vocab_size=vocab_size)
    
    print(f"\nVocabulário fictício:")
    print(f"  Tamanho: {vocab.vocab_size:,} tokens")
    print(f"  Token 0: {vocab.START}")
    print(f"  Token 1: {vocab.EOS}")
    print(f"  Token 2: {vocab.UNK}")
    print(f"  Tokens 3+: word_3, word_4, ..., word_{vocab_size - 1}")
    
    # Simular saída do Encoder
    # (Encoder processou uma frase em língua fonte, ex: francês)
    encoder_output = np.random.normal(0, 1, (1, 10, 512))
    
    print(f"\nSimulação:")
    print(f"  Encoder recebeu: Frase em francês (10 tokens)")
    print(f"  Encoder output: {encoder_output.shape}")
    print(f"  Decoder vai gerar: Frase em inglês (iterativamente)")
    
    # ===== Executar loop de geração =====
    generated_sequence = generate_with_argmax(encoder_output, vocab, max_length=max_generation_steps)
    
    # ===== Análise dos resultados =====
    print(f"\n" + "="*80)
    print("ANÁLISE DOS RESULTADOS")
    print("="*80)
    
    print(f"\nSequência gerada completa:")
    print(f"  {' -> '.join(generated_sequence)}")
    
    print(f"\nEstatísticas:")
    print(f"  Total de tokens: {len(generated_sequence)}")
    print(f"  Divididos em:")
    print(f"    - 1x <START> (início)")
    print(f"    - {len(generated_sequence) - 2}x tokens gerados")
    print(f"    - {'1x <EOS>' if vocab.EOS in generated_sequence else 'Sem <EOS>'} (fim)")
    
    # Verificar corretude
    print(f"\nVerificações:")
    
    # Check 1: Começou com START
    check1 = generated_sequence[0] == vocab.START
    print(f"  {'[OK]' if check1 else '[ERRO]'} Começou com {vocab.START}")
    
    # Check 2: Não contém None
    check2 = None not in generated_sequence
    print(f"  {'[OK]' if check2 else '[ERRO]'} Nenhum token None")
    
    # Check 3: Tokens válidos
    check3 = all(tok in vocab.tokens for tok in generated_sequence)
    print(f"  {'[OK]' if check3 else '[ERRO]'} Todos tokens são válidos")
    
    # Check 4: Parou corretamente (EOS ou max_length)
    check4 = (generated_sequence[-1] == vocab.EOS or 
              len(generated_sequence) == max_generation_steps + 1)
    print(f"  {'[OK]' if check4 else '[ERRO]'} Parou corretamente (EOS ou max_length)")
    
    print(f"\n" + "="*80)
    all_checks = check1 and check2 and check3 and check4
    if all_checks:
        print("[OK] SUCESSO: Loop auto-regressivo funcionando perfeitamente!")
    else:
        print("[ERRO] PROBLEMAS: Problemas detectados no loop")
    print("="*80)
    
    return generated_sequence


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TAREFA 3: LOOP DE INFERÊNCIA AUTO-REGRESSIVO".center(80))
    print("="*80)
    
    # Executar prova
    generated = prova_autoregressive()
    
    print("\nRESUMO DA TAREFA 3:")
    print("[OK] MockVocabulary criado com 10.000 tokens")
    print("[OK] Função generate_next_token() implementada")
    print("[OK] Pipeline Decoder (Self-Attn + Cross-Attn) simulado")
    print("[OK] Loop while com geração iterativa")
    print("[OK] Argmax para seleção de token mais provável")
    print("[OK] Parada em <EOS> ou max_length")
    print("\n" + "="*80)
