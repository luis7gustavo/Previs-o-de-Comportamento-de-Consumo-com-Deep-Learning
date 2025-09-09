import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Attention, Concatenate, GlobalAveragePooling1D, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import warnings
import holidays

warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Carrega os dados de um arquivo Excel, limpa, pré-processa e extrai features.
    """
    df = pd.read_excel(file_path, sheet_name='Online Retail')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print("   - Removendo outliers e limpando os dados...")
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df.dropna(subset=['CustomerID'], inplace=True)
    
    # Filtra valores para remover outliers extremos
    df = df[df['Quantity'] > 0]
    df = df[df['Quantity'] < 9000]
    df = df[df['UnitPrice'] > 0]
    df = df[df['UnitPrice'] < 3000]
    
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    
    # Agrega as vendas por dia
    daily_sales = df.set_index('InvoiceDate')['TotalPrice'].resample('D').sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    daily_sales['y'] = daily_sales['y'].fillna(0)
    
    # Cria features de data
    daily_sales['day_of_week'] = daily_sales['ds'].dt.dayofweek
    daily_sales['week_of_year'] = daily_sales['ds'].dt.isocalendar().week.astype(int)
    
    # Adiciona feature de feriado no Reino Unido
    uk_holidays = holidays.UnitedKingdom(years=daily_sales['ds'].dt.year.unique())
    daily_sales['is_holiday'] = daily_sales['ds'].isin(uk_holidays).astype(int)
    
    return daily_sales

def create_multivariate_sequences(data, target_col_index, sequence_length):
    """
    Cria sequências de dados para o modelo de série temporal.
    """
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length, target_col_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_and_train_attention_lstm(df, sequence_length=60, test_period_days=60):
    """
    Constrói, treina e avalia o modelo Bi-LSTM com Attention.
    """    
    dates = df['ds']
    features_df = df.drop('ds', axis=1)
    
    # Normaliza os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features_df)

    # Divide os dados em treino e teste
    train_size = len(scaled_data) - test_period_days
    train_data = scaled_data[:train_size]
    # Garante que o test_data tenha o tamanho correto para criar as sequências
    test_data = scaled_data[train_size - sequence_length:]

    X_train, y_train = create_multivariate_sequences(train_data, 0, sequence_length)
    X_test, y_test = create_multivariate_sequences(test_data, 0, sequence_length)
    
    num_features = X_train.shape[2]
    
    # --- Construção do Modelo ---
    latent_dim = 100
    
    encoder_inputs = Input(shape=(sequence_length, num_features))
    
    encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    encoder_outputs, fwd_h, fwd_c, bwd_h, bwd_c = encoder_lstm(encoder_inputs)
    
    state_h = Concatenate()([fwd_h, bwd_h])
    
    attention_layer = Attention()
    attention_result = attention_layer([encoder_outputs, encoder_outputs])
    context_vector = GlobalAveragePooling1D()(attention_result)

    decoder_concat_input = Concatenate(axis=-1)([state_h, context_vector])
    
    dense_output = Dense(100, activation='relu')(decoder_concat_input)
    dense_output = Dropout(0.2)(dense_output)
    dense_output = Dense(50, activation='relu')(dense_output)
    output = Dense(1)(dense_output)

    model = Model(inputs=encoder_inputs, outputs=output)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # --- Treinamento do Modelo ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=150,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # --- Avaliação no Conjunto de Teste ---
    predictions_scaled = model.predict(X_test)
    
    # Inverte a escala para obter os valores originais
    dummy_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_predictions[:, 0] = predictions_scaled.ravel()
    predictions_original = scaler.inverse_transform(dummy_predictions)[:, 0]

    dummy_y_test = np.zeros((len(y_test), num_features))
    dummy_y_test[:, 0] = y_test.ravel()
    y_test_original = scaler.inverse_transform(dummy_y_test)[:, 0]

    # Calcula métricas de erro para o conjunto de teste
    rmse_test = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mae_test = mean_absolute_error(y_test_original, predictions_original)
    r2_test = r2_score(y_test_original, predictions_original)
    print(f"\nAvaliação (Teste): RMSE={rmse_test:.2f}, MAE={mae_test:.2f}, R²={r2_test:.2f}")

    # --- Gráfico 1: Desempenho no Conjunto de Teste ---
    plt.figure(figsize=(15, 6))
    plot_dates_test = dates.iloc[-len(y_test_original):]
    plt.plot(plot_dates_test, y_test_original, color='blue', label='Vendas Reais (Teste)')
    plt.plot(plot_dates_test, predictions_original, color='red', linestyle='--', label='Previsão (Teste)')
    plt.title('Comparação entre Vendas Reais e Previsão (Conjunto de Teste)')
    plt.xlabel('Data')
    plt.ylabel('Vendas Totais (£)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # --- Avaliação e Gráfico para o Dataset Completo ---
    print("\n   - Gerando previsões e métricas para o dataset completo...")

    # Previsões para os dados de treino
    train_predictions_scaled = model.predict(X_train)
    dummy_train_predictions = np.zeros((len(train_predictions_scaled), num_features))
    dummy_train_predictions[:, 0] = train_predictions_scaled.ravel()
    train_predictions_original = scaler.inverse_transform(dummy_train_predictions)[:, 0]
    
    # Combina as previsões de treino e teste
    full_predictions = np.concatenate([train_predictions_original, predictions_original])

    # Pega os valores reais de y para treino e inverte a escala
    dummy_y_train = np.zeros((len(y_train), num_features))
    dummy_y_train[:, 0] = y_train.ravel()
    y_train_original = scaler.inverse_transform(dummy_y_train)[:, 0]
    
    # Combina os valores reais de treino e teste
    full_actuals = np.concatenate([y_train_original, y_test_original])
    
    # --- NOVA SEÇÃO: Cálculo das Métricas para o Dataset Completo ---
    rmse_full = np.sqrt(mean_squared_error(full_actuals, full_predictions))
    mae_full = mean_absolute_error(full_actuals, full_predictions)
    r2_full = r2_score(full_actuals, full_predictions)
    print(f"Avaliação (Completo): RMSE={rmse_full:.2f}, MAE={mae_full:.2f}, R²={r2_full:.2f}")

    # Seleciona as datas correspondentes para o gráfico completo
    # O primeiro valor previsto corresponde à data após o fim da primeira sequência
    plot_dates_full = dates.iloc[sequence_length:sequence_length + len(full_actuals)]

    # --- Gráfico 2: Desempenho no Dataset Completo (Treino + Teste) ---
    plt.figure(figsize=(15, 6))
    plt.plot(plot_dates_full, full_actuals, color='blue', label='Vendas Reais (Dataset Completo)')
    plt.plot(plot_dates_full, full_predictions, color='orange', linestyle='--', label='Previsão (Dataset Completo)')
    # Linha vertical para separar treino e teste
    plt.axvline(x=plot_dates_test.iloc[0], color='green', linestyle='--', label='Início do Período de Teste')
    plt.title('Comparação entre Vendas Reais e Previsão (Dataset Completo)')
    plt.xlabel('Data')
    plt.ylabel('Vendas Totais (£)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Função principal para executar o processo.
    """
    # Certifique-se de que o arquivo 'Online Retail.xlsx' está no mesmo diretório
    try:
        file_path = 'Online Retail.xlsx'
        daily_data = load_and_prepare_data(file_path)
        build_and_train_attention_lstm(daily_data)
    except FileNotFoundError:
        print(f"Erro: O arquivo '{file_path}' não foi encontrado.")
        print("Por favor, faça o download do arquivo ou verifique o caminho.")

if __name__ == "__main__":
    main()
