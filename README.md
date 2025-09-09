# Previsão de Vendas Online com Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
</p>

## 📊 Sobre o Projeto

Este projeto demonstra conclusivamente que a previsão de séries temporais de varejo com alta volatilidade atinge um nível de excelência através da sinergia entre uma arquitetura de Deep Learning avançada e uma engenharia de features criteriosa.

O objetivo principal foi desenvolver um modelo preditivo de alta performance para o comportamento de consumo diário, utilizando um dataset de varejo online (_Online Retail_). A solução evoluiu de modelos clássicos para uma arquitetura sofisticada de Deep Learning, alcançando um notável nível de precisão.


## Estrutura do Projeto

```
previsao_de_VendasOnline/
├── config/             # Configurações do projeto e modelo
├── data/               # Dados brutos e processados
├── notebooks/          # Jupyter notebooks para análises
├── src/                # Código fonte principal
│   ├── data/           # Scripts para carregamento e processamento de dados
│   ├── features/       # Scripts para engenharia de features
│   ├── models/         # Scripts para treinar e fazer previsões
│   └── utils/          # Utilitários como logging e visualização
├── tests/              # Testes unitários
├── main.py             # Script principal para execução do pipeline
├── requirements.txt    # Dependências do projeto
└── setup.py            # Script de instalação do pacote
```

## 🔬 Metodologia

O pipeline do projeto foi estruturado em três etapas principais:

### 1️⃣ Pré-processamento e Engenharia de Features
- Higienização do dataset bruto
- Transformação em série temporal
- Criação de features temporais:
  - `day_of_week`
  - `week_of_year`
  - `is_holiday`

### 2️⃣ Análise Comparativa
Diferentes modelos foram avaliados em dois cenários:

| Modelo | Cenário 1: Features Mínimas | Cenário 2: Features Enriquecidas |
|--------|----------------------------|----------------------------------|
| Árvore de Decisão | ✓ | ✓ |
| Random Forest | ✓ | ✓ |
| XGBoost | ✓ | ✓ |
| LSTM | ✓ | ✓ |
| Arquitetura Final | ✓ | ✓ |

- **Cenário 1**: Utilizando apenas os dados brutos agregados por dia
- **Cenário 2**: Incorporando o contexto temporal da engenharia de features

### 3️⃣ Arquitetura Final
Rede Neural Recorrente Bidirecional (Bi-LSTM) com um Mecanismo de Atenção:
- Processamento da sequência de dados em ambas as direções
- Foco dinâmico nos dias mais influentes para a previsão
  

## Requisitos

- Python 3.8+
- TensorFlow 2.13+
- pandas, numpy, scikit-learn e outras dependências listadas em `requirements.txt`


## 📝 Como Usar

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/luis7gustavo/previsao_de_VendasOnline.git
cd previsao_de_VendasOnline
pip install -r requirements.txt
```

### Treinamento do Modelo

Para treinar um novo modelo com os dados padrão:

```bash
python main.py --mode train
```

Para especificar um arquivo de dados personalizado:

```bash
python main.py --mode train --data caminho/para/dados.csv
```

### Geração de Previsões

Para gerar previsões usando um modelo já treinado:

```bash
python main.py --mode predict --data caminho/para/novos_dados.csv --steps 30
```


## Estrutura de Dados

O arquivo de dados de entrada deve conter pelo menos:
- Uma coluna de data (configurável em `config/config.yaml`)
- Uma coluna de valores de vendas (configurável em `config/config.yaml`)


## Arquitetura do Modelo

A solução utiliza uma arquitetura LSTM (Long Short-Term Memory) multicamada com:
- Camadas de dropout para evitar overfitting
- Normalização de dados
- Early stopping para otimização do treinamento


## Principais Técnicas de Engenharia de Features

- Extração de componentes temporais (dia da semana, mês, etc.)
- Indicadores de feriados e eventos especiais
- Features cíclicas para capturar sazonalidade
- Valores defasados (lags) e estatísticas móveis (rolling)

  
## 👨‍💻 Autoria

[Pedro Rebello](https://github.com/PedroRebello1)
[Luis Gustavo](https://github.com/luis7gustavo)

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
