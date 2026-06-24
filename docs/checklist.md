# Checklist do Projeto — PCO213 Aprendizado de Máquina

**Tema:** Predição de Ocorrência de Chuva no Dia Seguinte  
**Dataset:** Rain in Australia / weatherAUS (Kaggle)  
**Variável-alvo:** `RainTomorrow` (classificação binária supervisionada)

---

## Status das Etapas

| Etapa | Descrição | Status |
|-------|-----------|--------|
| 1 | Estrutura inicial, config, data_loader, README, .gitignore | ✅ Concluída |
| 2 | EDA: figuras, estatísticas, dicionário de dados preenchido | ⏳ Aguardando |
| 3 | Pré-processamento: limpeza estrutural + ColumnTransformer | ⏳ Aguardando |
| 4 | Modelagem: CV, treino, tuning (Grid/Randomized) | ⏳ Aguardando |
| 5 | Comparação final, main.py, figures 7–10, best_model.joblib | ⏳ Aguardando |
| 6 | Docs: artigo_estrutura.md, slides_roteiro.md, referencias.md | ⏳ Aguardando |
| 7 | README final completo | ⏳ Aguardando |

---

## Etapa 1 — O que foi criado

- `data/{raw,processed}/` com `.gitkeep`
- `notebooks/`, `src/`, `outputs/{figures,tables,reports}/`, `models/`, `docs/` com `.gitkeep`
- `.gitignore` (ignora raw/, processed/, *.joblib, caches; preserva outputs/)
- `requirements.txt` (Python 3.12 compatível)
- `README.md` (objetivo, estrutura, fallback Kaggle, instruções manuais)
- `src/__init__.py`
- `src/config.py` (paths, constantes, RANDOM_STATE=42, TARGET, TEST_SIZE, CV_FOLDS, etc.)
- `src/data_loader.py` (download, load_raw, basic_inspect, build_data_dictionary)
- `docs/checklist.md` (este arquivo)

---

## Próximas etapas — Etapa 2 (EDA)

- [ ] Criar `src/eda.py` com funções de estatísticas descritivas e geração de figuras
- [ ] Criar `src/visualization.py` com helpers de plotagem e `save_fig()`
- [ ] Criar `notebooks/01_eda.ipynb` (camada visual, importa `src/`)
- [ ] Gerar e salvar as figuras 1–6:
  1. Distribuição de `RainTomorrow` (desbalanceamento da classe)
  2. Percentual de valores ausentes por coluna
  3. Histogramas das principais variáveis numéricas
  4. Boxplots das principais variáveis numéricas
  5. Heatmap de correlação
  6. Relação entre `RainToday` e `RainTomorrow`
- [ ] Preencher campos `decisao` e `justificativa` no dicionário de dados
- [ ] Analisar outliers (método IQR)
- [ ] Decidir empiricamente quais colunas manter/remover com base no % de ausentes e relevância meteorológica

---

## Lembretes críticos — Aplicam a todas as etapas

### Data leakage — NUNCA fazer
- ❌ Ajustar `SimpleImputer`, `StandardScaler` ou `OneHotEncoder` no dataset inteiro
- ❌ Usar dados do teste no treinamento de qualquer forma
- ✅ Split treino/teste **antes** de qualquer ajuste de transformador
- ✅ Imputação, encoding e scaling somente dentro de `Pipeline`/`ColumnTransformer`
- ✅ Conjunto de teste: usado **apenas** na avaliação final

### Métricas — Usar todas
- accuracy, precision, recall, F1-score, ROC-AUC
- Matriz de confusão, curva ROC, classification report
- Curva Precision-Recall (se possível)
- **Foco especial em Recall**: falso negativo (prever "não chove" quando chove) tem custo prático maior

### Artigo científico — Proibido
- ❌ Código-fonte no corpo do artigo
- ✅ Usar equações, fluxogramas, pseudocódigo, diagrama de pipeline
- ✅ Toda figura deve ter título, legenda e estar citada/explicada no texto

### Resultados — Regras absolutas
- ❌ Inventar qualquer número de métrica antes da execução real
- ❌ Escrever accuracy/F1/recall/ROC-AUC sem ter rodado os modelos
- ✅ Números de desempenho só existem após execução e avaliação real

### SVM — Custo computacional
- `SVC` com kernel RBF é inviável em ~142k linhas (complexidade O(n²))
- Usar `LinearSVC(class_weight="balanced")`
- Treinar apenas em amostra estratificada do treino (`SVM_SAMPLE_SIZE = 30_000`)
- Avaliar no mesmo conjunto de teste dos demais modelos
- Documentar essa decisão na metodologia do artigo

### Reprodutibilidade
- `random_state=42` em todos os pontos
- `train_test_split(test_size=0.2, stratify=y, random_state=42)`
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`

---

## Estrutura de arquivos esperada ao final do projeto

```
machine-learning-project/
├── data/raw/weatherAUS.csv                  (não versionado)
├── data/processed/weatherAUS_processed.csv  (não versionado; regenerável)
├── notebooks/01_eda.ipynb
├── notebooks/02_modeling.ipynb
├── notebooks/03_results_interpretation.ipynb
├── src/config.py
├── src/data_loader.py
├── src/eda.py
├── src/preprocessing.py
├── src/modeling.py
├── src/evaluation.py
├── src/visualization.py
├── outputs/figures/          (≥10 figuras salvas)
├── outputs/tables/metrics_comparison.csv
├── outputs/tables/data_dictionary.csv
├── models/best_model.joblib  (não versionado; regenerável)
├── docs/artigo_estrutura.md
├── docs/slides_roteiro.md
├── docs/referencias.md
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```
