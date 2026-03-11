# -*- coding: utf-8 -*-
"""
LABORATÓRIO 2: IMPLEMENTANDO O DECODER DO TRANSFORMER

Coordenador das três tarefas principais:
1. Tarefa 1: Máscara Causal (Look-Ahead Mask)
2. Tarefa 2: Cross-Attention (Ponte Encoder-Decoder)
3. Tarefa 3: Loop de Inferência Auto-Regressivo

Execute este arquivo para rodar todas as tarefas em sequência:
    python main.py

Ou execute tarefas individuais:
    python tarefa1_mascara_causal.py
    python tarefa2_cross_attention.py
    python tarefa3_loop_autoregressive.py

Autor: Laboratório de Processamento de Linguagem Natural
"""

import numpy as np
import sys

# Importar as funções de cada tarefa
from tarefa1_mascara_causal import prova_mascara_causal
from tarefa2_cross_attention import prova_cross_attention
from tarefa3_loop_autoregressive import prova_autoregressive


def print_header(title: str):
    """Imprime cabeçalho formatado."""
    width = 80
    print("\n" + "="*width)
    print(title.center(width))
    print("="*width + "\n")


def print_section_separator(title: str):
    """Imprime separador de seção."""
    print("\n" + "="*80)
    print(title)
    print("="*80)


def main():
    """Executa todas as três tarefas do laboratório."""
    
    print_header("LABORATÓRIO 2: IMPLEMENTANDO O DECODER DO TRANSFORMER")
    
    print("Objetivos de Aprendizagem:")
    print("  1. Dominar a álgebra linear do mascaramento causal")
    print("  2. Implementar a ponte Encoder-Decoder (Cross-Attention)")
    print("  3. Simular um loop de geração auto-regressiva\n")
    
    # =========================================================================
    # TAREFA 1: MÁSCARA CAUSAL
    # =========================================================================
    print_section_separator("EXECUTANDO TAREFA 1: MÁSCARA CAUSAL")
    try:
        causal_mask, attention_weights = prova_mascara_causal(seq_len=5)
        print("\n[OK] Tarefa 1 concluída com sucesso!")
    except Exception as e:
        print(f"\n[ERROR] Erro na Tarefa 1: {e}")
        return False
    
    # =========================================================================
    # TAREFA 2: CROSS-ATTENTION
    # =========================================================================
    print_section_separator("EXECUTANDO TAREFA 2: CROSS-ATTENTION")
    try:
        attention_output, cross_weights = prova_cross_attention()
        print("\n[OK] Tarefa 2 concluída com sucesso!")
    except Exception as e:
        print(f"\n[ERROR] Erro na Tarefa 2: {e}")
        return False
    
    # =========================================================================
    # TAREFA 3: LOOP AUTO-REGRESSIVO
    # =========================================================================
    print_section_separator("EXECUTANDO TAREFA 3: LOOP AUTO-REGRESSIVO")
    try:
        generated = prova_autoregressive()
        print("\n[OK] Tarefa 3 concluída com sucesso!")
    except Exception as e:
        print(f"\n[ERROR] Erro na Tarefa 3: {e}")
        return False
    
    # =========================================================================
    # RESUMO FINAL
    # =========================================================================
    print_section_separator("RESUMO EXECUTADO COM SUCESSO")
    
    print("""
[COMPLETO] TAREFA 1: Máscara Causal (Look-Ahead Mask)
  - Impede que posições futuras recebam atenção
  - Probabilidades futuras = 0.0 após Softmax
  - Modelo não consegue "olhar para o futuro"

[COMPLETO] TAREFA 2: Cross-Attention (Ponte Encoder-Decoder)
  - Integração estrutural entre duas redes neurais
  - Query vem do Decoder, Keys/Values do Encoder
  - Decoder pode atender completamente ao Encoder
  - Sem máscara causal (acesso irrestrito)

[COMPLETO] TAREFA 3: Loop de Inferência Auto-Regressivo
  - Geração iterativa token-por-token
  - Uso de argmax para seleção greedy
  - Parada em <EOS> ou limite de comprimento
  - Pipeline: Self-Attn + Cross-Attn + FFN + Softmax
    """)
    
    print("="*80)
    print("\nTodas as três tarefas foram implementadas e validadas com sucesso!\n")
    
    return True


if __name__ == "__main__":
    np.random.seed(42)  # Reprodutibilidade
    success = main()
    sys.exit(0 if success else 1)
