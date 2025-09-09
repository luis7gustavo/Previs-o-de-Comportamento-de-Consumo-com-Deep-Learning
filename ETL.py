import pandas as pd
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import re
import getpass

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'database': 'bd_comprebem01'
}
NOME_ARQUIVO_ENTRADA = 'dados_brutos_comprebem.xlsx'

def extrair_dados(caminho_arquivo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_vendas = pd.read_excel(caminho_arquivo, sheet_name='Vendas')
    df_produtos = pd.read_excel(caminho_arquivo, sheet_name='Estoque de Produtos')
    print("Dados extraídos")
    return df_vendas, df_produtos

def limpar_preco(preco_str: str) -> float:
    if isinstance(preco_str, (int, float)):
        return float(preco_str)
    preco_limpo = re.sub(r'[^\d,]', '', str(preco_str)).replace(',', '.')
    return float(preco_limpo) if preco_limpo else 0.0

def padronizar_categorias(df_produtos: pd.DataFrame) -> pd.DataFrame:
    mapa_categorias = {
        'eletronico': 'Eletrônicos',
        'smartphones': 'Eletrônicos',
        'cozinha': 'Casa e Cozinha',
        'eletrodomésticos': 'Eletrodomésticos'
    }
    df_produtos['Categoria'] = df_produtos['Categoria'].str.lower().replace(mapa_categorias)
    df_produtos['Categoria'] = df_produtos['Categoria'].str.title()
    return df_produtos

def transformar_dados(df_vendas: pd.DataFrame, df_produtos: pd.DataFrame) -> dict:
   
    df_produtos = padronizar_categorias(df_produtos)
    df_produtos['Preco Base'] = df_produtos['Preco Base'].apply(limpar_preco)
    df_produtos['Descricao'] = df_produtos['Descricao'].fillna('Descrição não disponível')

    categorias_unicas = df_produtos[['Categoria']].drop_duplicates().rename(columns={'Categoria': 'NomeCategoria'})
    categorias_unicas['Descricao'] = 'Categoria de produtos'

    df_clientes = df_vendas[['Nome do Cliente', 'Email do Cliente', 'Endereco de Entrega']].copy()
    df_clientes.drop_duplicates(subset=['Email do Cliente'], inplace=True)
    df_clientes.rename(columns={
        'Nome do Cliente': 'Nome',
        'Email do Cliente': 'Email',
        'Endereco de Entrega': 'Endereco'
    }, inplace=True)
    df_clientes['SenhaHash'] = 'senha_mock_hash_123'
    df_clientes['Cidade'] = 'Cidade Fictícia'
    df_clientes['Estado'] = 'SP'
    df_clientes['CEP'] = '12345-678'
    df_clientes['Telefone'] = '(11) 98765-4321'

    df_vendas['Valor Unitario'] = df_vendas['Valor Unitario'].apply(limpar_preco)
    df_vendas['Data da Venda'] = pd.to_datetime(df_vendas['Data da Venda'], format='%d-%m-%Y %H:%M')
    df_vendas['Status'] = df_vendas['Status'].str.title()

    df_vendas['Valor Total Item'] = df_vendas['Quantidade'] * df_vendas['Valor Unitario']
    pedidos_agg = df_vendas.groupby('ID do Pedido').agg(
        ValorTotal=('Valor Total Item', 'sum'),
        DataPedido=('Data da Venda', 'first'),
        StatusPedido=('Status', 'first'),
        EmailCliente=('Email do Cliente', 'first'),
        EnderecoEntrega=('Endereco de Entrega', 'first')
    ).reset_index()

    df_itens_pedido = df_vendas[[
        'ID do Pedido',
        'SKU do Produto',
        'Quantidade',
        'Valor Unitario'
    ]].copy()
    df_itens_pedido.rename(columns={
        'ID do Pedido': 'PedidoID_Original',
        'SKU do Produto': 'SKU',
        'Valor Unitario': 'PrecoUnitario'
    }, inplace=True)

    return {
        "categorias": categorias_unicas,
        "produtos": df_produtos,
        "clientes": df_clientes,
        "pedidos": pedidos_agg,
        "itens_pedido": df_itens_pedido
    }

def conectar_banco(config: dict):
    try:
        password = getpass.getpass(f"Digite a senha para o usuário '{config['user']}': ")
        conexao = mysql.connector.connect(password=password, **config)
        if conexao.is_connected():
            print("Conexão estabelecida.")
            return conexao
    except Error as e:
        print(f"ERRO de Conexão: {e}")
        raise
    return None

def carregar_dados(conexao, cursor, df: pd.DataFrame, nome_tabela: str, campos_map: dict) -> dict:
    mapa_ids = {}
    chave_original = next(iter(campos_map))

    cols_db = list(campos_map.values())
    placeholders = ", ".join(["%s"] * len(cols_db))
    query = f"INSERT INTO {nome_tabela} ({', '.join(cols_db)}) VALUES ({placeholders})"

    for _, row in df.iterrows():
        valores = tuple(row[col_df] for col_df in campos_map.keys())
        cursor.execute(query, valores)
        novo_id = cursor.lastrowid
        mapa_ids[row[chave_original]] = novo_id

    return mapa_ids

def pipeline_carga(conexao, dados_transformados: dict):
    cursor = None
    try:
        cursor = conexao.cursor()
        conexao.start_transaction()

        mapa_cat = carregar_dados(conexao, cursor,
                                  dados_transformados['categorias'], 'Categorias',
                                  {'NomeCategoria': 'NomeCategoria', 'Descricao': 'Descricao'})

        mapa_cli = carregar_dados(conexao, cursor,
                                  dados_transformados['clientes'], 'Clientes',
                                  {'Email': 'Email', 'Nome': 'Nome', 'Endereco': 'Endereco', 'SenhaHash': 'SenhaHash',
                                   'Cidade': 'Cidade', 'Estado': 'Estado', 'CEP': 'CEP', 'Telefone': 'Telefone'})

        df_prod = dados_transformados['produtos'].copy()
        df_prod['CategoriaID'] = df_prod['Categoria'].map(mapa_cat)
        mapa_prod = carregar_dados(conexao, cursor,
                                   df_prod, 'Produtos',
                                   {'SKU': 'ProdutoID', 'Nome do Produto': 'Nome', 'Descricao': 'Descricao',
                                    'Preco Base': 'Preco', 'CategoriaID': 'CategoriaID', 'Estoque': 'QuantidadeEstoque'})

        df_ped = dados_transformados['pedidos'].copy()
        df_ped['ClienteID'] = df_ped['EmailCliente'].map(mapa_cli)
        mapa_ped = carregar_dados(conexao, cursor,
                                  df_ped, 'Pedidos',
                                  {'ID do Pedido': 'PedidoID', 'ClienteID': 'ClienteID', 'DataPedido': 'DataPedido',
                                   'StatusPedido': 'StatusPedido', 'ValorTotal': 'ValorTotal', 'EnderecoEntrega': 'EnderecoEntrega'})

        df_itens = dados_transformados['itens_pedido'].copy()
        df_itens['PedidoID'] = df_itens['PedidoID_Original'].map(mapa_ped)
        df_itens['ProdutoID'] = df_itens['SKU'].map(mapa_prod)
        carregar_dados(conexao, cursor,
                       df_itens, 'Itens_Pedido',
                       {'PedidoID': 'PedidoID', 'ProdutoID': 'ProdutoID', 'Quantidade': 'Quantidade', 'PrecoUnitario': 'PrecoUnitario'})

        conexao.commit()
    except Error as e:
        print(f"ERRO durante a carga de dados: {e}")
        if conexao:
            conexao.rollback()
        raise
    finally:
        if cursor:
            cursor.close()

def main():
    conexao = None
    try:
        df_vendas, df_produtos = extrair_dados(NOME_ARQUIVO_ENTRADA)
        dados_transformados = transformar_dados(df_vendas, df_produtos)

        conexao = conectar_banco(DB_CONFIG)
        if conexao:
            pipeline_carga(conexao, dados_transformados)

    except Exception as e:
        print("\nPipeline ETL falhou.")
    finally:
        if conexao and conexao.is_connected():
            conexao.close()
            print("\nConexão encerrada.")

if __name__ == '__main__':
    main()

# https://www.canva.com/design/DAGrBzyJPXA/PDAWe6h4uHtJInpLVFoXVQ/edit?utm_content=DAGrBzyJPXA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton