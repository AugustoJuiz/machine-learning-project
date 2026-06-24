# Roteiro de Slides — Apresentação (até 15 minutos)
# Predição de Chuva no Dia Seguinte com Machine Learning

> **Estrutura:** 14 slides | **Tempo:** ~15 minutos (≈1 min/slide)
> Para cada slide: conteúdo visual sugerido + fala resumida.

---

## Slide 1 — Título e Equipe

**Conteúdo visual:**
- Título: *Predição de Ocorrência de Chuva no Dia Seguinte Utilizando Técnicas de Aprendizado de Máquina*
- Subtítulo: PCO213 — Aprendizado de Máquina e Mineração de Dados | UNIFEI | [Semestre]
- Nomes dos integrantes
- Imagem: mapa da Austrália ou ícone de chuva

**Fala sugerida:**
> "Bom dia/tarde. Nosso trabalho final propõe a aplicação de técnicas de aprendizado de máquina
> para prever se choverá no dia seguinte, usando dados meteorológicos históricos da Austrália."

---

## Slide 2 — Problema e Motivação

**Conteúdo visual:**
- Ícones ou imagens representando: agricultura, defesa civil, transporte, turismo
- Frase de impacto: "Prever chuva pode salvar vidas e evitar prejuízos econômicos."
- Pergunta norteadora: *"Dado o clima observado hoje, choverá amanhã?"*

**Fala sugerida:**
> "A previsão de chuva é crítica em várias áreas. Agricultores precisam saber se vão colher ou
> não. A defesa civil precisa emitir alertas. Nosso modelo busca responder: com base no clima
> de hoje, choverá amanhã?"

---

## Slide 3 — Dataset

**Conteúdo visual:**
- Logo/referência do Kaggle
- Tabela resumo: ~142k observações | ~23 colunas | 10 anos de dados | ~50 estações australianas
- Variável-alvo: `RainTomorrow` (Yes/No)
- Fig. 1: Distribuição de RainTomorrow (gráfico de pizza)

**Fala sugerida:**
> "Utilizamos o dataset Rain in Australia, com [PREENCHER] observações de estações espalhadas
> pela Austrália. A variável-alvo é RainTomorrow — binária — indicando se haverá chuva amanhã.
> Como podemos ver, o dataset é desbalanceado: cerca de [PREENCHER]% dos dias não chove."

---

## Slide 4 — Pipeline Geral da Solução

**Conteúdo visual:**
- Diagrama de fluxo (Mermaid ou versão visual):

```
Dataset bruto → EDA → Limpeza → Split → Pipeline sklearn → CV → Tuning → Avaliação → Melhor modelo
```

- Destaque: a linha de "sem data leakage" separando as etapas de treino e teste

**Fala sugerida:**
> "Nosso pipeline segue as boas práticas de ciência de dados: primeiro exploramos os dados,
> depois os limpamos, dividimos em treino e teste, aplicamos pré-processamento dentro de
> um Pipeline sklearn — garantindo que nenhuma informação do teste vaze para o treino —
> e por fim treinamos, avaliamos e selecionamos o melhor modelo."

---

## Slide 5 — Análise Exploratória: Dados Ausentes e Variáveis

**Conteúdo visual:**
- Fig. 2: Gráfico de valores ausentes (barras horizontais)
- Tabela resumida do dicionário de dados (5–6 variáveis mais relevantes)

**Fala sugerida:**
> "Durante a EDA identificamos colunas com alto percentual de dados ausentes.
> Colunas acima de 40% foram candidatas à remoção. As variáveis meteorológicas mais
> promissoras para previsão de chuva são umidade, pressão, nebulosidade e vento."

---

## Slide 6 — EDA: Distribuições e Correlações

**Conteúdo visual:**
- Fig. 4: Boxplots de Humidity3pm e Pressure3pm agrupados por RainTomorrow (detalhe)
- Fig. 5: Heatmap de correlação (versão menor ou recorte das variáveis-chave)
- Fig. 7: Scatter plots de pares meteorológicos coloridos por RainTomorrow
  (ex.: Humidity3pm × Pressure3pm mostrando separação entre classes)
- Destaque: correlações com RainTomorrow

**Fala sugerida:**
> "Os boxplots mostram que dias com chuva amanhã tendem a ter maior umidade e menor pressão
> hoje à tarde. O heatmap de correlação confirma essa relação. Os scatter plots (Fig. 7)
> evidenciam visualmente a separação entre classes: pontos laranjas — dias que choverão —
> concentram-se em faixas de alta umidade e baixa pressão. Também observamos que RainToday
> é um forte preditor de RainTomorrow (Fig. 6)."

---

## Slide 7 — Pré-processamento

**Conteúdo visual:**
- Diagrama do ColumnTransformer:

```
Variáveis numéricas  → Imputer(mediana) → StandardScaler
Variáveis categóricas → Imputer(moda)   → OneHotEncoder
```

- Diagrama Pipeline: `ColumnTransformer → Classifier`
- Nota: "Ajuste apenas no treino — sem data leakage"

**Fala sugerida:**
> "O pré-processamento foi implementado dentro de um Pipeline sklearn, garantindo que a
> imputação, normalização e codificação sejam aprendidas apenas no treino. Valores ausentes
> numéricos são substituídos pela mediana; categóricos, pela moda. Categóricas são codificadas
> com OneHotEncoding."

---

## Slide 8 — Modelos Avaliados

**Conteúdo visual:**
- Tabela com 5 modelos:

| Modelo | Tipo | Observação |
|---|---|---|
| DummyClassifier | Baseline | Sempre prevê a classe mais frequente |
| Logistic Regression | Linear | Baseline interpretável |
| Decision Tree | Árvore | Regras explícitas |
| Random Forest | Ensemble | Mais robusto |
| LinearSVC | SVM | Amostra de 30k do treino |

- Destaque: `class_weight='balanced'` em todos os classificadores reais

**Fala sugerida:**
> "Avaliamos cinco modelos. O DummyClassifier serve como referência mínima. Os demais usam
> class_weight='balanced' para compensar o desbalanceamento. O LinearSVC foi treinado em uma
> amostra de 30.000 exemplos do treino por limitações computacionais, mas avaliado no mesmo
> conjunto de teste."

---

## Slide 9 — Validação Cruzada e Métricas

**Conteúdo visual:**
- Tabela de CV (F1 média ± dp, ROC-AUC média ± dp): [PREENCHER após execução]
- Fórmulas compactas de F1 e ROC-AUC
- Destaque: "Foco em Recall — falso negativo tem custo maior"

**Fala sugerida:**
> "A validação cruzada com 5 folds estratificados estimou o desempenho dos modelos sem tocar
> no conjunto de teste. Priorizamos Recall — prever 'não chove' quando de fato chove é o erro
> mais custoso. Os resultados indicaram [PREENCHER] como modelo mais promissor."

---

## Slide 10 — Resultados Comparativos (Fig 11)

**Conteúdo visual:**
- Fig. 11: Gráfico de barras comparando Accuracy, Precision, Recall, F1, ROC-AUC entre modelos
- Destaque visual no melhor modelo

**Fala sugerida:**
> "Após avaliação final no conjunto de teste, o [PREENCHER] obteve os melhores resultados.
> Comparado ao baseline: F1 de [PREENCHER] versus [PREENCHER] do DummyClassifier,
> uma melhora de [PREENCHER] pontos percentuais."

---

## Slide 11 — Melhor Modelo: Matriz de Confusão (Fig 8)

**Conteúdo visual:**
- Fig. 8: Matriz de confusão do melhor modelo (versão colorida)
- Explicação dos quadrantes: TN, FP, FN, TP com ícones de chuva/sol
- Métricas: Precision = [PREENCHER], Recall = [PREENCHER], F1 = [PREENCHER]

**Fala sugerida:**
> "A matriz de confusão do [PREENCHER] mostra que o modelo acertou [PREENCHER]% dos dias
> chuvosos (Recall) e [PREENCHER]% das previsões de chuva estavam corretas (Precision).
> Os [PREENCHER] falsos negativos representam os casos de maior risco prático."

---

## Slide 12 — Curvas ROC e Importância das Variáveis (Fig 9 + Fig 12)

**Conteúdo visual:**
- Fig. 9: Curvas ROC de todos os modelos (metade do slide)
- Fig. 12: Top 10 features mais importantes — Random Forest (outra metade)

**Fala sugerida:**
> "As curvas ROC (Fig. 9) confirmam a superioridade do [PREENCHER] com AUC = [PREENCHER].
> Quanto às variáveis, [PREENCHER] e [PREENCHER] foram as mais importantes — o que faz
> sentido meteorológico: alta umidade à tarde e quedas de pressão indicam frentes chuvosas."

---

## Slide 13 — Discussão e Limitações

**Conteúdo visual:**
- Dois blocos: "O que funcionou" vs. "Limitações"
- O que funcionou: ensemble supera linear; class_weight compensa desbalanceamento; Humidity3pm e Pressure3pm como preditores chave
- Limitações: dados exclusivos da Austrália; sem modelagem temporal; colunas com muitos ausentes; SVM em amostra reduzida

**Fala sugerida:**
> "O pipeline funcionou bem para o contexto proposto. Mas há limitações: os modelos ignoram
> a dependência temporal entre dias consecutivos, e o dataset é restrito à Austrália. Além disso,
> colunas como Sunshine e Evaporation tinham muitos ausentes, o que pode ter reduzido o potencial
> preditivo."

---

## Slide 14 — Conclusão e Trabalhos Futuros

**Conteúdo visual:**
- Conclusão principal: "[PREENCHER] foi o melhor modelo — F1=[PREENCHER], ROC-AUC=[PREENCHER]"
- Lista de trabalhos futuros (ícones): XGBoost, séries temporais (LSTM), por localidade, calibração, API/dashboard
- Repositório: `D:\Unifei\MachineLearning\machine-learning-project`
- Agradecimento

**Fala sugerida:**
> "Em conclusão, o [PREENCHER] se mostrou o melhor modelo para previsão de chuva no dia
> seguinte, com F1 de [PREENCHER] e ROC-AUC de [PREENCHER].
> Como trabalhos futuros, pretendemos explorar XGBoost, modelos de séries temporais
> e um dashboard interativo. Obrigado! Estamos à disposição para perguntas."

---

## Tempo estimado por slide

| Slide | Conteúdo | Tempo |
|---|---|---|
| 1 | Título e equipe | 0:30 |
| 2 | Problema e motivação | 1:00 |
| 3 | Dataset | 1:00 |
| 4 | Pipeline geral | 1:30 |
| 5 | EDA — ausentes e variáveis | 1:00 |
| 6 | EDA — distribuições e correlações | 1:00 |
| 7 | Pré-processamento | 1:00 |
| 8 | Modelos avaliados | 1:00 |
| 9 | CV e métricas | 1:00 |
| 10 | Resultados comparativos | 1:30 |
| 11 | Matriz de confusão | 1:00 |
| 12 | Curvas ROC e importâncias | 1:00 |
| 13 | Discussão e limitações | 1:00 |
| 14 | Conclusão e trabalhos futuros | 1:30 |
| **Total** | | **~15:00** |
