# Estrutura do Artigo Científico
# Predição de Ocorrência de Chuva no Dia Seguinte Utilizando Técnicas de Aprendizado de Máquina

> **Instrução de uso:** Este arquivo é um roteiro textual para o artigo final.
> Substitua as marcações `[PREENCHER]` pelos valores reais obtidos na execução do pipeline.
> O artigo deve ter **no máximo 6 páginas**, sem código-fonte no corpo do texto.
> Toda figura deve ser citada e explicada no texto com referência a `outputs/figures/`.

---

## Título

**Predição de Ocorrência de Chuva no Dia Seguinte Utilizando Técnicas de Aprendizado de Máquina**

---

## Autores

[Nome do integrante 1] — UNIFEI
[Nome do integrante 2] — UNIFEI
(PCO213 — Aprendizado de Máquina e Mineração de Dados — Semestre [XX/XXXX])

---

## Resumo (Abstract)

A previsão de precipitação atmosférica constitui um problema de relevância prática em múltiplos
domínios, incluindo agricultura, defesa civil, transporte e planejamento urbano. O presente trabalho
propõe o desenvolvimento e a comparação de modelos de aprendizado de máquina para a classificação
supervisionada binária da ocorrência de chuva no dia seguinte, utilizando o conjunto de dados
*Rain in Australia* (weatherAUS), disponível publicamente no repositório Kaggle. Foram aplicadas
as seguintes técnicas: `DummyClassifier` (baseline), Regressão Logística, Árvore de Decisão,
Random Forest e Suporte a Vetores de Máquina Linear (LinearSVC). O pipeline contempla análise
exploratória de dados, tratamento de valores ausentes, codificação de variáveis categóricas,
padronização numérica e validação cruzada estratificada com cinco folds. Os modelos foram avaliados
pelas métricas *accuracy*, *precision*, *recall*, F1-score e ROC-AUC, com ênfase no *recall*,
dada a relevância prática do falso negativo neste contexto. O modelo [PREENCHER] obteve os
melhores resultados, alcançando F1 = [PREENCHER] e ROC-AUC = [PREENCHER] no conjunto de teste,
superando o baseline em [PREENCHER] pontos percentuais de F1.

**Palavras-chave:** aprendizado de máquina; classificação supervisionada; previsão de chuva;
Random Forest; validação cruzada.

---

## 1. Introdução

A previsão de precipitação atmosférica desempenha papel fundamental em diversas áreas de atividade
humana. Na agricultura, a antecipação de eventos chuvosos permite o planejamento de colheitas,
irrigação e aplicação de defensivos agrícolas. Na defesa civil, possibilita a emissão de alertas
para prevenção de enchentes e deslizamentos. No setor de transportes e logística, contribui para
a otimização de rotas e prevenção de acidentes. No turismo e na gestão de eventos ao ar livre,
orienta decisões operacionais e reduz perdas econômicas.

Os métodos tradicionais de previsão meteorológica baseiam-se em modelos numéricos de atmosfera
que demandam infraestrutura computacional robusta e dados de alta resolução espacial. O aprendizado
de máquina apresenta-se como uma abordagem complementar, capaz de extrair padrões preditivos
a partir de dados históricos de estações meteorológicas, com menor custo computacional e maior
facilidade de atualização.

O presente trabalho formula a previsão de chuva como um problema de **classificação supervisionada
binária**: dado um conjunto de observações meteorológicas do dia atual, predizer se haverá
precipitação no dia seguinte (`RainTomorrow = Yes` ou `No`). O conjunto de dados utilizado é o
*Rain in Australia* (weatherAUS), que reúne observações diárias de diversas estações meteorológicas
australianas ao longo de aproximadamente dez anos.

O objetivo principal é desenvolver, avaliar e comparar múltiplos algoritmos de aprendizado de
máquina, identificando aquele que melhor equilibra as métricas de desempenho relevantes para
o contexto de previsão de chuva, com atenção especial ao *recall* (sensibilidade).

---

## 2. Fundamentação Teórica

### 2.1 Aprendizado Supervisionado e Classificação Binária

O aprendizado supervisionado consiste no treinamento de um modelo a partir de pares
entrada-saída `(x_i, y_i)`, onde `x_i` é o vetor de atributos e `y_i` é o rótulo da classe
[REFERÊNCIA]. Na classificação binária, `y_i ∈ {0, 1}`. O objetivo é aprender uma função
`f: X → Y` que generalize para dados não vistos.

### 2.2 Regressão Logística

A Regressão Logística modela a probabilidade da classe positiva por meio da função sigmoide:

```
P(y=1 | x) = σ(w^T x + b) = 1 / (1 + e^{-(w^T x + b)})
```

onde `w` é o vetor de pesos aprendido por maximização da verossimilhança. É um modelo linear
interpretável, adequado como baseline comparativo [REFERÊNCIA].

### 2.3 Árvore de Decisão

A Árvore de Decisão particiona recursivamente o espaço de atributos minimizando a impureza de
Gini ou a entropia informacional em cada nó. O critério de parada controla a profundidade máxima
e o número mínimo de amostras por folha, evitando sobreajuste [REFERÊNCIA].

### 2.4 Random Forest

O Random Forest é um método *ensemble* que constrói `T` árvores de decisão em amostras bootstrap
do conjunto de treino (*bagging*), utilizando um subconjunto aleatório de atributos em cada
divisão (*feature randomness*). A predição final é obtida por votação majoritária. Esta estratégia
reduz a variância em relação a uma árvore única [REFERÊNCIA].

A importância de uma variável é estimada pela redução média de impureza nos nós em que ela é
utilizada, integrada sobre todas as árvores da floresta.

### 2.5 Máquina de Vetores de Suporte (SVM)

A SVM busca o hiperplano de margem máxima que separa as classes no espaço de características.
O parâmetro `C` controla o equilíbrio entre maximizar a margem e minimizar erros de classificação.
O `LinearSVC` implementa a formulação linear, eficiente para conjuntos de dados grandes [REFERÊNCIA].

### 2.6 Validação Cruzada Estratificada

A validação cruzada com `k` folds estratificados divide os dados em `k` partições, preservando
a proporção das classes em cada fold. O desempenho é estimado pela média e desvio padrão da
métrica de interesse sobre os `k` experimentos [REFERÊNCIA].

### 2.7 Métricas de Avaliação

Para um classificador binário, as métricas derivadas da matriz de confusão são:

```
Accuracy  = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F1-score  = 2 · (Precision · Recall) / (Precision + Recall)
```

A curva ROC traça TPR (Recall) versus FPR para diferentes limiares de classificação.
A área sob a curva (ROC-AUC) mede a capacidade discriminativa do modelo de forma independente
do limiar [REFERÊNCIA].

### 2.8 Dados Desbalanceados e Custo do Falso Negativo

O dataset weatherAUS apresenta desbalanceamento de classes (~78% `No`, ~22% `Yes`). Em cenários
de previsão de chuva, o **falso negativo** (prever `No` quando ocorre `Yes`) tem custo prático
superior ao falso positivo, pois leva à ausência de preparação para eventos chuvosos. Por essa
razão, o *recall* é priorizado como métrica de interesse, e `class_weight='balanced'` é empregado
nos classificadores sensíveis ao desbalanceamento.

---

## 3. Metodologia

### 3.1 Fonte e Descrição do Dataset

O conjunto de dados *Rain in Australia* (weatherAUS) foi obtido do repositório público Kaggle
(slug: `jsphyg/weather-dataset-rattle-package`). Contém aproximadamente [PREENCHER] observações
diárias coletadas em [PREENCHER] estações meteorológicas australianas no período de [PREENCHER]
a [PREENCHER].

**Variável-alvo:** `RainTomorrow` — indica se houve precipitação no dia seguinte
(`Yes` → 1; `No` → 0).

O **Dicionário de Dados** (Tabela 1, salvo em `outputs/tables/data_dictionary.csv`) apresenta
todos os atributos, seus tipos, percentual de valores ausentes e relevância meteorológica.

### 3.2 Análise Exploratória de Dados (EDA)

A EDA foi conduzida em sete etapas, cada uma resultando em uma figura salva em `outputs/figures/`:

- **Fig. 1** — Distribuição de `RainTomorrow`: evidencia o desbalanceamento de classes
  (~[PREENCHER]% `No` versus ~[PREENCHER]% `Yes`), fundamentando o uso de `class_weight='balanced'`.
- **Fig. 2** — Percentual de valores ausentes por coluna: colunas com mais de 40% de ausentes
  foram identificadas como candidatas à remoção (ver Tabela 1).
- **Fig. 3** — Histogramas das principais variáveis numéricas: permite identificar assimetria
  e concentração de valores.
- **Fig. 4** — Boxplots agrupados por `RainTomorrow`: variáveis com separação visual clara entre
  classes (ex.: `Humidity3pm`, `Pressure3pm`) são fortes preditores.
- **Fig. 5** — Heatmap de correlação de Pearson: analisa multicolinearidade e correlação com o alvo.
- **Fig. 6** — Relação entre `RainToday` e `RainTomorrow`: dias com chuva apresentam maior
  probabilidade de chuva no dia seguinte.
- **Fig. 7** — Scatter plots de pares meteorológicos coloridos por `RainTomorrow`
  (`Humidity3pm × Pressure3pm`, `Humidity3pm × Temp3pm`, `WindGustSpeed × Pressure3pm`,
  `Rainfall × Humidity3pm`): confirma separação visual entre classes para variáveis-chave,
  evidenciando que alta umidade à tarde e baixa pressão caracterizam dias chuvosos.

A análise de outliers pelo método IQR (Tabela 2, em `outputs/tables/outlier_analysis.csv`)
identificou valores extremos em [PREENCHER] das colunas. Optou-se por não remover outliers,
pois podem representar eventos meteorológicos extremos de alta relevância preditiva.

### 3.3 Pré-processamento

O pipeline de pré-processamento foi implementado com `Pipeline` e `ColumnTransformer` do
scikit-learn, assegurando que todos os transformadores fossem ajustados **exclusivamente** no
conjunto de treino, eliminando data leakage.

**Limpeza estrutural** (antes do split, salva em `data/processed/`):
- Remoção de linhas com `RainTomorrow` ausente ([PREENCHER] linhas removidas).
- Conversão de `RainToday` e `RainTomorrow`: `Yes → 1`, `No → 0`.
- Extração de `Month` (mês) e `Season` (estação australiana) a partir da coluna `Date`.
- Remoção de colunas com percentual de ausentes acima de 40%: [PREENCHER] colunas removidas
  (ver Tabela 1 para justificativas).

> **Nota metodológica:** a decisão de remover colunas por alto percentual de ausentes é uma
> operação estrutural e determinística, baseada na taxa de missingness do dataset completo
> e na relevância meteorológica da variável — sem uso da variável-alvo nem de qualquer
> estatística do conjunto de teste. Não configura data leakage. Imputação, normalização
> e codificação permanecem exclusivamente dentro do `Pipeline`/`ColumnTransformer`,
> ajustados apenas no conjunto de treino.

**Dentro do Pipeline/ColumnTransformer** (ajustados apenas no treino):
- **Variáveis numéricas:** `SimpleImputer(strategy='median')` → `StandardScaler()`.
- **Variáveis categóricas:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`.

### 3.4 Divisão Treino/Teste e Validação Cruzada

O dataset processado foi dividido em treino (80%) e teste (20%) com estratificação por classe
(`train_test_split(test_size=0.2, stratify=y, random_state=42)`). O conjunto de teste foi
isolado e utilizado **apenas** na avaliação final.

A validação cruzada empregou `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`,
reportando média e desvio padrão de F1 e ROC-AUC nos cinco folds.

### 3.5 Modelos Avaliados

| Modelo | Parâmetros principais | Observações |
|---|---|---|
| DummyClassifier | `strategy='most_frequent'` | Baseline majoritário |
| LogisticRegression | `class_weight='balanced'`, `max_iter=1000` | Baseline interpretável |
| DecisionTreeClassifier | `class_weight='balanced'` | Tuning: GridSearch |
| RandomForestClassifier | `class_weight='balanced'`, `n_estimators=100` | Tuning: RandomizedSearch |
| LinearSVC (calibrado) | `class_weight='balanced'`, `max_iter=2000` | Amostra estratificada de 30.000 amostras do treino |

**Nota sobre o LinearSVC:** o custo computacional do `SVC` com kernel RBF é O(n²), inviável
para ~[PREENCHER] amostras de treino. Optou-se pelo `LinearSVC` encapsulado em
`CalibratedClassifierCV` (3 folds internos) para obter probabilidades e ROC-AUC.
O treinamento utilizou amostra estratificada de 30.000 amostras do treino; a avaliação
final ocorreu no mesmo conjunto de teste dos demais modelos.

### 3.6 Otimização de Hiperparâmetros

- **Árvore de Decisão:** `GridSearchCV` com grade: `max_depth ∈ {3,5,8,None}`,
  `min_samples_split ∈ {2,10,20}`, `min_samples_leaf ∈ {1,5,10}`, `criterion ∈ {gini, entropy}`.
- **Random Forest:** `RandomizedSearchCV` (`n_iter=20`) com `n_estimators ∈ {50,100,200}`,
  `max_depth ∈ {5,10,20,None}`, `min_samples_split ∈ {2,5,10}`, `min_samples_leaf ∈ {1,3,5}`.
- **Regressão Logística:** `GridSearchCV` sobre `C ∈ {0.01, 0.1, 1.0, 10.0}`.

### 3.7 Métricas e Critério de Seleção

Métricas calculadas no conjunto de teste: *accuracy*, *precision*, *recall*, F1-score,
ROC-AUC e *average precision* (AP). O critério de seleção do melhor modelo prioriza
**F1-score** (excluindo o baseline), com ROC-AUC como desempate e *recall* como
métrica de interesse prático.

### 3.8 Pipeline Geral

O diagrama abaixo representa o fluxo completo do pipeline implementado:

```mermaid
flowchart LR
  A[Dataset bruto\nweatherAUS.csv] --> B[EDA\nFig 1-7]
  B --> C[Limpeza estrutural\ndata/processed/]
  C --> D[Split treino/teste\n80% / 20% estratificado]
  D --> E[Pipeline sklearn\nColumnTransformer]
  E --> F[Validacao cruzada\nStratifiedKFold k=5]
  F --> G[Tuning\nGrid/RandomizedSearch]
  G --> H[Treinamento\n5 modelos]
  H --> I[Avaliacao no teste\nFig 7-10]
  I --> J[Selecao do\nmelhor modelo]
  J --> K[Relatorio\ne Slides]
```

---

## 4. Experimentos e Resultados

### 4.1 Características do Dataset

Após limpeza estrutural, o dataset contém [PREENCHER] linhas e [PREENCHER] colunas.
A distribuição da variável-alvo é: `No` = [PREENCHER] ([PREENCHER]%) e `Yes` = [PREENCHER]
([PREENCHER]%), confirmando o desbalanceamento (Fig. 1).

O percentual de valores ausentes variou de [PREENCHER]% a [PREENCHER]% entre as colunas
(Fig. 2). As colunas [PREENCHER] foram removidas por excederem o limiar de 40% de ausentes
(ver Tabela 1 para justificativas).

A análise de correlação (Fig. 5) revelou correlação positiva de `Humidity3pm` com `RainTomorrow`
(r ≈ [PREENCHER]) e correlação negativa de `Pressure3pm` (r ≈ [PREENCHER]), alinhando-se com
o conhecimento meteorológico.

### 4.2 Validação Cruzada

**Tabela 3 — Resultados de Validação Cruzada (StratifiedKFold k=5, conjunto de treino):**

| Modelo | F1 (média ± dp) | ROC-AUC (média ± dp) | Recall (média) |
|---|---|---|---|
| Baseline | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| LogReg | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| DecTree | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| RandomForest | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| SVM | [PREENCHER] | [PREENCHER] | [PREENCHER] |

### 4.3 Avaliação Final no Conjunto de Teste

**Tabela 4 — Métricas no conjunto de teste (salva em `outputs/tables/metrics_comparison.csv`):**

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| LogReg | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| DecTree | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| RandomForest | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |
| SVM | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] | [PREENCHER] |

A Fig. 8 apresenta a matriz de confusão do modelo [PREENCHER]. A Fig. 9 exibe as curvas ROC
de todos os modelos, evidenciando a superioridade de [PREENCHER] com área [PREENCHER].
A Fig. 10 mostra a curva Precision-Recall do melhor modelo. A Fig. 11 compara visualmente
as métricas entre modelos.

### 4.4 Importância das Variáveis

A Fig. 12 apresenta as variáveis mais relevantes identificadas pelo Random Forest e pelos
coeficientes da Regressão Logística. As variáveis [PREENCHER] e [PREENCHER] foram as mais
importantes, consistentes com o conhecimento meteorológico: [PREENCHER].

---

## 5. Discussão

### 5.1 Desempenho Comparativo

O modelo [PREENCHER] obteve o melhor desempenho geral, com F1 = [PREENCHER] e ROC-AUC =
[PREENCHER] no conjunto de teste, superando o baseline em [PREENCHER] pontos percentuais
de F1. [DESCREVER: por que esse modelo superou os demais?]

A Regressão Logística, apesar de ser um modelo linear simples, demonstrou desempenho competitivo
com F1 = [PREENCHER], confirmando que relações lineares entre variáveis meteorológicas e a
ocorrência de chuva são informativamente relevantes.

A Árvore de Decisão, após tuning, obteve F1 = [PREENCHER], [MAIOR/MENOR] que a Regressão
Logística, [INTERPRETAR].

### 5.2 Trade-off Precision × Recall

O uso de `class_weight='balanced'` priorizou o *recall*, aumentando a sensibilidade dos modelos
à classe minoritária (`RainTomorrow = Yes`). Este comportamento é adequado ao contexto de
aplicação: o custo de não alertar sobre chuva é maior que o de um alerta falso.

[DESCREVER O TRADE-OFF OBSERVADO NOS RESULTADOS REAIS]

### 5.3 Variáveis de Maior Relevância

As variáveis [PREENCHER] e [PREENCHER] foram as mais importantes, [INTERPRETAR no contexto
meteorológico].

### 5.4 Limitações

- Os dados são exclusivos da Austrália; a generalização para outras regiões requer reavaliação.
- Alto percentual de ausentes em colunas como `Sunshine` e `Evaporation` pode ter reduzido o
  poder preditivo.
- A abordagem ignora a estrutura temporal dos dados (dias consecutivos), tratando cada
  observação como independente.
- O `LinearSVC` foi treinado em amostra de 30.000 observações, podendo não representar
  plenamente a distribuição completa do treino.

---

## 6. Conclusão

O presente trabalho desenvolveu um pipeline completo de ciência de dados para a previsão
de chuva no dia seguinte utilizando o dataset *Rain in Australia*. Foram implementados e
comparados cinco algoritmos de classificação supervisionada, com avaliação criteriosa por
métricas adequadas ao contexto de dados desbalanceados.

O modelo [PREENCHER] apresentou o melhor desempenho, alcançando F1 = [PREENCHER] e
ROC-AUC = [PREENCHER]. As variáveis `Humidity3pm` e `Pressure3pm` destacaram-se como
os preditores mais relevantes, alinhando-se ao conhecimento meteorológico sobre umidade
relativa e sistemas frontais.

O pipeline implementado é reproduzível: basta executar `python main.py` com o arquivo
`weatherAUS.csv` posicionado em `data/raw/` para regenerar todos os resultados, figuras
e o modelo exportado.

### Trabalhos Futuros

- Aplicação de modelos de séries temporais (LSTM, Temporal Fusion Transformer) para capturar
  dependência temporal entre dias consecutivos.
- Avaliação de modelos por localidade (estação meteorológica), dado que padrões climáticos
  variam geograficamente na Austrália.
- Incorporação de algoritmos de *gradient boosting* (XGBoost, LightGBM, CatBoost).
- Calibração probabilística dos classificadores (Platt Scaling, Regressão Isotônica).
- Desenvolvimento de API REST ou dashboard interativo (Streamlit/Gradio) para uso prático.
- Uso de dados mais recentes via API do Bureau of Meteorology da Austrália.

---

## Referências

*(Ver docs/referencias.md para lista completa formatada)*
