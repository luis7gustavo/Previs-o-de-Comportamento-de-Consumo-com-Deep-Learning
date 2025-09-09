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

## 🎯 Objetivo

Prever o comportamento de consumo utilizando:
- Arquitetura avançada de redes neurais
- Contextualização de negócio (feriados, sazonalidade, etc.)
- Decodificação de padrões complexos com grande acurácia

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

## 🛠️ Tecnologias Utilizadas

<details>
<summary>Clique para expandir</summary>

- **Python**: Linguagem principal para desenvolvimento
- **Pandas**: Ingestão, higienização e transformação dos dados
- **NumPy**: Manipulação de arrays multidimensionais para operações do TensorFlow
- **Scikit-learn**: Pré-processamento (normalização) e avaliação de modelos
- **TensorFlow & Keras**: Construção, treinamento e validação da arquitetura de rede neural
- **Matplotlib**: Visualização dos resultados para análise qualitativa
- **Holidays**: Utilizado na engenharia de features para identificar feriados

</details>

## 📈 Resultados e Conclusão

A introdução de features contextuais causou uma transformação radical nos resultados:

- O modelo Bi-LSTM + Atenção alcançou um **coeficiente de determinação (R²) de 0.964** no conjunto de teste
- Isto significa que o modelo explica **96.4% da variabilidade dos dados**

<p align="center">
  <img src="https://via.placeholder.com/800x400?text=Gráfico+de+Resultados" alt="Resultados do Modelo" width="600">
</p>

## 💡 Conclusões

Este projeto valida a hipótese de que a ciência de dados moderna pode gerar valor tangível e uma vantagem competitiva significativa para a tomada de decisões de negócio, como:

- Otimização de inventário
- Planejamento de marketing
- Alocação de recursos

A sinergia entre uma arquitetura de Deep Learning avançada e uma engenharia de features criteriosa é fundamental para atingir excelência na previsão de séries temporais de varejo com alta volatilidade.

## 📝 Como Usar

```bash
# Clone o repositório
git clone https://github.com/luis7gustavo/previsao_de_VendasOnline.git

# Entre no diretório
cd previsao_de_VendasOnline

# Instale as dependências
pip install -r requirements.txt

# Execute o notebook principal
jupyter notebook main_analysis.ipynb
```

## 👨‍💻 Autoria

[Pedro Rebello](https://github.com/PedroRebello1)
[Luis Gustavo](https://github.com/luis7gustavo)

## 📄 Licença

Este projeto está sob a licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.
