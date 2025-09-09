# -*- coding: utf-8 -*-

import pandas as pd
from faker import Faker
import random
from datetime import datetime

# Inicializa o Faker para gerar dados em português do Brasil
fake = Faker('pt_BR')

def gerar_dados_produtos(num_produtos: int) -> pd.DataFrame:
    """
    Gera uma lista de produtos com inconsistências para simular dados brutos.
    """
    print(f"Gerando {num_produtos} produtos com dados inconsistentes...")
    
    # Categorias com inconsistências propositais (maiúsculas/minúsculas, sinônimos)
    categorias_sujas = ['Eletrônicos', 'eletronico', 'Casa e Cozinha', 'Cozinha', 'smartphones', 'Eletrodomésticos']
    
    produtos = []
    for i in range(num_produtos):
        # Simula inconsistência no nome do produto
        nome_produto = f"Produto {chr(65 + i)} {fake.word().capitalize()}"
        
        # Simula preços como string e com vírgula como separador decimal
        preco = f"R$ {random.uniform(50.0, 2500.0):.2f}".replace('.', ',')
        
        # Simula dados faltantes (descrição)
        descricao = fake.text(max_nb_chars=80) if random.random() > 0.2 else None
        
        produto = {
            'SKU': f"CB-{'0'*(5-len(str(i)))}{i}", # Código único do produto
            'Nome do Produto': nome_produto,
            'Categoria': random.choice(categorias_sujas),
            'Preco Base': preco,
            'Estoque': random.randint(0, 150),
            'Descricao': descricao
        }
        produtos.append(produto)
        
    return pd.DataFrame(produtos)


def gerar_dados_vendas(num_vendas: int, df_produtos: pd.DataFrame) -> pd.DataFrame:
    """
    Gera uma lista de vendas denormalizada, simulando uma exportação de sistema.
    """
    print(f"Gerando dados para {num_vendas} vendas...")
    vendas = []
    
    # Lista de status com variações
    status_pedido = ['entregue', 'processando', 'Enviado', 'Cancelado']
    
    for i in range(num_vendas):
        # Dados do cliente são repetidos para cada item do mesmo pedido
        nome_cliente = fake.name()
        email_cliente = fake.email()
        endereco_cliente = fake.address().replace('\n', ', ')
        data_pedido = fake.date_time_between(start_date='-2y', end_date='now')
        
        # Cada pedido pode ter de 1 a 4 itens
        num_itens_pedido = random.randint(1, 4)
        
        for _ in range(num_itens_pedido):
            # Seleciona um produto aleatório da lista de produtos
            produto_vendido = df_produtos.sample(1).to_dict('records')[0]
            
            venda = {
                'ID do Pedido': 1000 + i,
                'Data da Venda': data_pedido.strftime('%d-%m-%Y %H:%M'), # Formato de data não padrão
                'Nome do Cliente': nome_cliente,
                'Email do Cliente': email_cliente,
                'Endereco de Entrega': endereco_cliente,
                'SKU do Produto': produto_vendido['SKU'],
                'Nome do Produto Vendido': produto_vendido['Nome do Produto'],
                'Quantidade': random.randint(1, 3),
                'Valor Unitario': produto_vendido['Preco Base'], # Mantém o formato de string
                'Status': random.choice(status_pedido)
            }
            vendas.append(venda)
            
    return pd.DataFrame(vendas)


def main():
    """
    Função principal que orquestra a geração de dados e a criação da planilha Excel.
    """
    try:
        # Define a quantidade de dados a serem gerados
        total_produtos = 50
        total_pedidos = 200
        
        # Gera os DataFrames
        df_produtos = gerar_dados_produtos(total_produtos)
        df_vendas = gerar_dados_vendas(total_pedidos, df_produtos)
        
        # Define o nome do arquivo de saída
        nome_arquivo = 'dados_brutos_comprebem.xlsx'
        
        print(f"\nSalvando dados na planilha '{nome_arquivo}'...")
        
        # Usa o ExcelWriter para salvar múltiplos DataFrames em abas diferentes
        with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
            df_vendas.to_excel(writer, sheet_name='Vendas', index=False)
            df_produtos.to_excel(writer, sheet_name='Estoque de Produtos', index=False)
            
        print("-" * 50)
        print(f"✅ Planilha '{nome_arquivo}' criada com sucesso!")
        print(f"   -> Aba 'Vendas' contém {len(df_vendas)} registros.")
        print(f"   -> Aba 'Estoque de Produtos' contém {len(df_produtos)} registros.")
        print("-" * 50)

    except Exception as e:
        print(f"\n❌ Ocorreu um erro ao gerar ou salvar o arquivo: {e}")

if __name__ == '__main__':
    main()