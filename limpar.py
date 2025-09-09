# -*- coding: utf-8 -*-
"""
Script de Pré-Processamento de Dados.

Este script realiza a etapa de ETL (Extração, Transformação e Carga) uma única vez.
Ele carrega o dataset bruto, aplica toda a limpeza, remoção de outliers e
engenharia de features, e salva o resultado em um novo arquivo CSV limpo,
pronto para ser usado pelo modelo de previsão.

Para executar, instale as dependências:
pip install pandas openpyxl holidays
"""
import pandas as pd
import holidays
import warnings

warnings.filterwarnings('ignore')

def preprocess_and_save(input_path, output_path):
    """
    Carrega o dataset original, limpa, remove outliers, cria features temporais,
    e salva o resultado em um novo arquivo CSV.

    Args:
        input_path (str): Caminho para o arquivo original .xlsx.
        output_path (str): Caminho onde o arquivo .csv limpo será salvo.
    """
    try:
        print(f"1. Carregando o dataset original de '{input_path}'...")
        df = pd.read_excel(input_path, sheet_name='Online Retail')
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        print("2. Removendo outliers e limpando os dados...")
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
        df.dropna(subset=['CustomerID'], inplace=True)
        
        df = df[df['Quantity'] > 0]
        df = df[df['Quantity'] < 9000]
        df = df[df['UnitPrice'] > 0]
        df = df[df['UnitPrice'] < 3000]
        
        print("3. Agregando vendas por dia...")
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        daily_sales = df.set_index('InvoiceDate')['TotalPrice'].resample('D').sum().reset_index()
        daily_sales.columns = ['ds', 'y']
        daily_sales['y'] = daily_sales['y'].fillna(0)
        
        print("4. Criando features temporais avançadas...")
        daily_sales['day_of_week'] = daily_sales['ds'].dt.dayofweek
        daily_sales['week_of_year'] = daily_sales['ds'].dt.isocalendar().week.astype(int)
        
        uk_holidays = holidays.UnitedKingdom(years=daily_sales['ds'].dt.year.unique())
        daily_sales['is_holiday'] = daily_sales['ds'].isin(uk_holidays).astype(int)
        
        print(f"5. Salvando o dataset limpo em '{output_path}'...")
        daily_sales.to_csv(output_path, index=False)
        
        print("\nProcessamento concluído com sucesso!")
        print(f"O arquivo limpo e pronto para uso foi salvo em: {output_path}")
        
    except FileNotFoundError:
        print(f"   - ERRO: O arquivo de entrada '{input_path}' não foi encontrado.")
    except Exception as e:
        print(f"   - ERRO: Ocorreu um erro inesperado: {e}")

if __name__ == "__main__":
    input_file = 'Online Retail.xlsx'
    output_file = 'online_retail_cleaned.csv'
    preprocess_and_save(input_file, output_file)
