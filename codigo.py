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
    df = pd.read_excel(file_path, sheet_name='Online Retail')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    print("   - Removendo outliers e limpando os dados...")
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df.dropna(subset=['CustomerID'], inplace=True)
    
    df = df[df['Quantity'] > 0]
    df = df[df['Quantity'] < 9000]
    df = df[df['UnitPrice'] > 0]
    df = df[df['UnitPrice'] < 3000]
    
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    daily_sales = df.set_index('InvoiceDate')['TotalPrice'].resample('D').sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    daily_sales['y'] = daily_sales['y'].fillna(0)
    
    daily_sales['day_of_week'] = daily_sales['ds'].dt.dayofweek
    daily_sales['week_of_year'] = daily_sales['ds'].dt.isocalendar().week.astype(int)
    
    uk_holidays = holidays.UnitedKingdom(years=daily_sales['ds'].dt.year.unique())
    daily_sales['is_holiday'] = daily_sales['ds'].isin(uk_holidays).astype(int)
    
    return daily_sales

def create_multivariate_sequences(data, target_col_index, sequence_length):
    xs, ys = [], []
    for i in range(len(data) - sequence_length):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length, target_col_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_and_train_attention_lstm(df, sequence_length=60, test_period_days=60):    
    dates = df['ds']
    features_df = df.drop('ds', axis=1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features_df)

    train_size = len(scaled_data) - test_period_days
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size - sequence_length:]

    X_train, y_train = create_multivariate_sequences(train_data, 0, sequence_length)
    X_test, y_test = create_multivariate_sequences(test_data, 0, sequence_length)
    
    num_features = X_train.shape[2]
    
    latent_dim = 100
    
    encoder_inputs = Input(shape=(sequence_length, num_features))
    
    # Encoder Bidirecional para capturar melhor o contexto
    encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
    encoder_outputs, fwd_h, fwd_c, bwd_h, bwd_c = encoder_lstm(encoder_inputs)
    
    # Concatena os estados finais do encoder bidirecional
    state_h = Concatenate()([fwd_h, bwd_h])
    
    # A camada de atenção foca nas saídas do encoder bidirecional
    attention_layer = Attention()
    attention_result = attention_layer([encoder_outputs, encoder_outputs])
    context_vector = GlobalAveragePooling1D()(attention_result)

    # Concatena o contexto da atenção com os estados concatenados do encoder
    decoder_concat_input = Concatenate(axis=-1)([state_h, context_vector])
    
    # Camadas densas para a previsão final
    dense_output = Dense(100, activation='relu')(decoder_concat_input)
    dense_output = Dropout(0.2)(dense_output)
    dense_output = Dense(50, activation='relu')(dense_output)
    output = Dense(1)(dense_output)

    model = Model(inputs=encoder_inputs, outputs=output)
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=150,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    predictions_scaled = model.predict(X_test)
    
    dummy_predictions = np.zeros((len(predictions_scaled), num_features))
    dummy_predictions[:, 0] = predictions_scaled.ravel()
    predictions_original = scaler.inverse_transform(dummy_predictions)[:, 0]

    dummy_y_test = np.zeros((len(y_test), num_features))
    dummy_y_test[:, 0] = y_test.ravel()
    y_test_original = scaler.inverse_transform(dummy_y_test)[:, 0]

    rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
    mae = mean_absolute_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)
    print(f"Avaliação: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.2f}")

    plt.figure(figsize=(15, 6))
    plot_dates = dates.iloc[-len(y_test_original):]
    plt.plot(plot_dates, y_test_original, color='blue', label='Vendas Reais')
    plt.plot(plot_dates, predictions_original, color='red', linestyle='--', label='Previsão (Bi-LSTM + Attention)')
    plt.title('Comparação entre Vendas Reais e Previsão (Modelo Final)')
    plt.xlabel('Data')
    plt.ylabel('Vendas Totais (£)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    file_path = 'Online Retail.xlsx'
    daily_data = load_and_prepare_data(file_path)

    build_and_train_attention_lstm(daily_data)

if __name__ == "__main__":
    main()
