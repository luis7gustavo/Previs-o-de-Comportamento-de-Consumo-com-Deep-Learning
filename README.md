# Previsão de Vendas Online com Deep Learning

Este projeto demonstra conclusivamente que a previsão de séries temporais de varejo com alta volatilidade atinge um nível de excelência através da sinergia entre uma arquitetura de Deep Learning avançada e uma engenharia de features.

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

## Funcionalidades

- Carregamento e pré-processamento de dados de séries temporais de vendas
- Engenharia de features avançada para séries temporais
- Modelagem com arquitetura de Deep Learning (LSTM) para previsão
- Avaliação de modelos e visualizações
- Pipeline completo de treinamento e previsão

## Requisitos

- Python 3.8+
- TensorFlow 2.13+
- pandas, numpy, scikit-learn e outras dependências listadas em `requirements.txt`

## Instalação

Clone o repositório e instale as dependências:

```bash
git clone https://github.com/luis7gustavo/previsao_de_VendasOnline.git
cd previsao_de_VendasOnline
pip install -r requirements.txt
```

## Uso

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

## Autor

Luis Gustavo

## Licença

Este projeto está licenciado sob os termos da licença MIT.