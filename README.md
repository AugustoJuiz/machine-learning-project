# Predição de Chuva no Dia Seguinte — Rain in Australia

Projeto final da disciplina **PCO213 — Aprendizado de Máquina e Mineração de Dados**  
**UNIFEI** — Universidade Federal de Itajubá

---

## Objetivo

Desenvolver um pipeline completo e reproduzível de ciência de dados para **classificação
supervisionada binária**: prever se choverá no dia seguinte (`RainTomorrow = Yes/No`)
a partir de variáveis meteorológicas observadas no dia atual.

---

## Dataset

| Atributo       | Descrição |
|----------------|-----------|
| **Nome**       | Rain in Australia / weatherAUS |
| **Fonte**      | [Kaggle — jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) |
| **Arquivo**    | `weatherAUS.csv` |
| **Registros**  | ~142 000 observações diárias de estações meteorológicas australianas |
| **Alvo**       | `RainTomorrow` — binário: `Yes` (chove) / `No` (não chove) |

---

## Estrutura do projeto

```
machine-learning-project/
├── data/
│   ├── raw/              # weatherAUS.csv (não versionado — ver abaixo)
│   └── processed/        # CSV com limpeza estrutural (não versionado — regenerável)
├── notebooks/            # 01_eda, 02_modeling, 03_results_interpretation
├── src/                  # Módulos Python do pipeline
│   ├── config.py         # Configurações globais e paths
│   ├── data_loader.py    # Carregamento, inspeção e dicionário de dados
│   ├── eda.py            # Análise exploratória e geração de figuras
│   ├── preprocessing.py  # Limpeza estrutural e ColumnTransformer
│   ├── modeling.py       # Definição e treinamento dos modelos
│   ├── evaluation.py     # Métricas, gráficos de avaliação e comparação
│   └── visualization.py  # Helpers de plotagem e salvamento de figuras
├── outputs/
│   ├── figures/          # Figuras geradas (versionadas)
│   ├── tables/           # Tabelas CSV de resultados (versionadas)
│   └── reports/          # Relatórios intermediários
├── models/               # best_model.joblib (não versionado — regenerável)
├── docs/                 # Estrutura do artigo, roteiro de slides, referências
├── requirements.txt
├── README.md
├── .gitignore
└── main.py               # Orquestra o pipeline ponta a ponta
```

---

## Configuração do ambiente

### 1. Clonar / entrar na pasta do projeto

```bash
cd D:\Unifei\MachineLearning\machine-learning-project
```

### 2. Criar e ativar ambiente virtual (recomendado)

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## Como obter o dataset

O arquivo `data/raw/weatherAUS.csv` **não é versionado** no repositório (pode ter ~70 MB).
Há duas formas de obtê-lo:

---

### Opção A — Download automático via Kaggle API (conveniência opcional)

> O pipeline funciona **sem** a Kaggle API. Ela é apenas uma conveniência para baixar
> o arquivo automaticamente. Se o CSV já existir em `data/raw/`, o download é pulado.

**Passo 1 — Obter credenciais Kaggle**

1. Acesse [kaggle.com](https://www.kaggle.com) → seu perfil → **Settings** → **API** → **Create New Token**.
2. O arquivo `kaggle.json` será baixado. Ele contém seu `username` e `key`.

**Passo 2 — Posicionar o arquivo de credenciais**

```
Windows:  %USERPROFILE%\.kaggle\kaggle.json
Linux/macOS: ~/.kaggle/kaggle.json
```

No Windows (PowerShell):
```powershell
mkdir $env:USERPROFILE\.kaggle -Force
Copy-Item kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```

**Passo 3 — Baixar o dataset**

```bash
python -c "from src.data_loader import download_dataset; download_dataset()"
```

Ou simplesmente rode `python main.py` — o download ocorre automaticamente se o CSV não existir.

---

### Opção B — Download manual (sem Kaggle API)

1. Acesse: <https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>
2. Clique em **Download** (requer conta gratuita no Kaggle).
3. Extraia o arquivo `weatherAUS.csv` do ZIP baixado.
4. Coloque o arquivo em:

```
machine-learning-project/
└── data/
    └── raw/
        └── weatherAUS.csv    ← aqui
```

Após isso, o pipeline funciona normalmente. **Nenhuma configuração adicional é necessária.**

---

## Como regenerar `data/processed/`

O arquivo `data/processed/weatherAUS_processed.csv` contém apenas limpeza estrutural
(remoção de linhas com alvo nulo, variáveis de calendário, conversão Yes/No → 0/1).
Ele é gerado automaticamente pelo pipeline. Para regenerar:

```bash
python -c "from src.preprocessing import clean_data; from src.data_loader import load_raw; clean_data(load_raw())"
```

Ou rode `python main.py` — o processamento ocorre na sequência do pipeline.

---

## Como executar o pipeline completo

```bash
python main.py
```

O script executa em sequência:
1. Carregamento e inspeção do dataset.
2. Análise exploratória (EDA) — gera figuras em `outputs/figures/`.
3. Limpeza estrutural — gera `data/processed/weatherAUS_processed.csv`.
4. Treinamento, validação cruzada e tuning dos modelos.
5. Avaliação final no conjunto de teste.
6. Exportação de métricas, gráficos e do melhor modelo.

---

## Modelos implementados

| Modelo | Observação |
|---|---|
| DummyClassifier | Baseline majoritário |
| LogisticRegression | Baseline interpretável |
| DecisionTreeClassifier | Explicável, regras, importância |
| RandomForestClassifier | Ensemble robusto, importância |
| LinearSVC | SVM linear (amostra estratificada do treino por custo computacional) |

---

## Notas sobre reprodutibilidade

- `random_state=42` em todos os pontos aplicáveis.
- Split treino/teste antes de qualquer ajuste de transformador.
- Imputação, encoding e scaling somente dentro de `Pipeline`/`ColumnTransformer`.
- O conjunto de teste é usado **apenas** na avaliação final.
