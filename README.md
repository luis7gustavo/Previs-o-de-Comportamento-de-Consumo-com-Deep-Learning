Previsão de Comportamento de Consumo com Deep Learning e Engenharia de Features
Este projeto foi desenvolvido para criar um modelo preditivo de alta performance para o comportamento de consumo diário, utilizando um dataset de varejo online (Online Retail). A solução evoluiu de modelos clássicos para uma arquitetura sofisticada de Deep Learning, culminando em um modelo de notável precisão.

Visão Geral
O objetivo principal foi prever o comportamento de consumo, utilizando uma arquitetura avançada de redes neurais com o contexto de negócio correto (feriados, sazonalidade, etc.) para decodificar padrões complexos com grande acurácia. A metodologia inclui uma análise comparativa rigorosa entre diferentes algoritmos, em cenários com e sem engenharia de features, provando o impacto crucial de dados enriquecidos na performance do modelo.

Arsenal Tecnológico
A execução do projeto exigiu um conjunto de ferramentas e bibliotecas específicas:

Python: Linguagem principal para desenvolvimento.

Pandas: Ingestão, higienização e transformação dos dados.

NumPy: Manipulação de arrays multidimensionais para operações do TensorFlow.

Scikit-learn: Pré-processamento (normalização) e avaliação de modelos.

TensorFlow & Keras: Construção, treinamento e validação da arquitetura de rede neural.

Matplotlib: Visualização dos resultados para análise qualitativa.

Holidays: Utilizado na engenharia de features para identificar feriados.

Metodologia e Modelos
O pipeline do projeto foi estruturado em três etapas principais:

Pré-processamento e Engenharia de Features: O dataset bruto foi higienizado e transformado em uma série temporal. A etapa mais impactante foi a criação de features temporais, como day_of_week, week_of_year e is_holiday.

Análise Comparativa: Diferentes modelos (Árvore de Decisão, Random Forest, XGBoost, LSTM e a arquitetura final) foram avaliados em dois cenários:

Cenário 1 (Features Mínimas): Utilizando apenas os dados brutos agregados por dia.

Cenário 2 (Features Enriquecidas): Incorporando o contexto temporal da engenharia de features.

Arquitetura Final: O modelo de melhor performance foi uma Rede Neural Recorrente Bidirecional (Bi-LSTM) com um Mecanismo de Atenção. Essa arquitetura processa a sequência de dados em ambas as direções, permitindo que o modelo foque dinamicamente nos dias mais influentes para a previsão.

Resultados e Conclusão
A introdução de features contextuais causou uma transformação radical nos resultados. O modelo Bi-LSTM + Atenção teve um salto de performance, alcançando um coeficiente de determinação (R 
2
 ) de 0.964 no conjunto de teste, explicando 96.4% da variabilidade dos dados.

Este projeto demonstra que a previsão de séries temporais de varejo com alta volatilidade atinge um nível de excelência através da sinergia entre uma arquitetura de Deep Learning avançada e uma engenharia de features criteriosa. Os resultados validam a hipótese de que a ciência de dados moderna pode gerar valor tangível e uma vantagem competitiva significativa para a tomada de decisões de negócio, como otimização de inventário, planejamento de marketing e alocação de recursos.
