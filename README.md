# Predição de Chuva no Dia Seguinte — Rain in Australia

Projeto final da disciplina **PCO213 — Aprendizado de Máquina e Mineração de Dados**  
**UNIFEI** — Universidade Federal de Itajubá

## Objetivo

Desenvolver um pipeline completo e reproduzível de ciência de dados para **classificação
supervisionada binária**: prever se choverá no dia seguinte (RainTomorrow = Yes/No)
a partir de variáveis meteorológicas observadas no dia atual.

## Dataset

| Atributo       | Descrição |
|----------------|-----------|
| **Nome**       | Rain in Australia / weatherAUS |
| **Fonte**      | [Kaggle — jsphyg/weather-dataset-rattle-package](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) |
| **Arquivo**    | weatherAUS.csv |
| **Registros**  | ~142 000 observações diárias de estações meteorológicas australianas |
| **Alvo**       | RainTomorrow — binário: Yes (chove) / No (não chove) |

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
├── docs/                 # Relatório final, estrutura do artigo, referências
├── requirements.txt
├── README.md
├── .gitignore
└── main.py               # Orquestra o pipeline ponta a ponta
```

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

## Como obter o dataset

O arquivo data/raw/weatherAUS.csv **não é versionado** no repositório (pode ter ~70 MB).
Há duas formas de obtê-lo:

### Opção A — Download automático via Kaggle API (conveniência opcional)

> O pipeline funciona **sem** a Kaggle API. Ela é apenas uma conveniência para baixar
> o arquivo automaticamente. Se o CSV já existir em data/raw/, o download é pulado.

**Passo 1 — Obter credenciais Kaggle**

1. Acesse [kaggle.com](https://www.kaggle.com) → seu perfil → **Settings** → **API** → **Create New Token**.
2. O arquivo kaggle.json será baixado. Ele contém seu username e key.

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

### Opção B — Download manual (sem Kaggle API)

1. Acesse: <https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package>
2. Clique em **Download** (requer conta gratuita no Kaggle).
3. Extraia o arquivo weatherAUS.csv do ZIP baixado.
4. Coloque o arquivo em:

```
machine-learning-project/
└── data/
    └── raw/
        └── weatherAUS.csv    ← aqui
```

Após isso, o pipeline funciona normalmente. **Nenhuma configuração adicional é necessária.**

## Como regenerar data/processed/

O arquivo data/processed/weatherAUS_processed.csv contém apenas limpeza estrutural
(remoção de linhas com alvo nulo, variáveis de calendário, conversão Yes/No → 0/1).
Ele é gerado automaticamente pelo pipeline. Para regenerar:

```bash
python -c "from src.preprocessing import clean_data; from src.data_loader import load_raw; clean_data(load_raw())"
```

Ou rode `python main.py` — o processamento ocorre na sequência do pipeline.

## Como executar o pipeline completo

**Windows (PowerShell) — recomendado:**
```powershell
$env:PYTHONUTF8="1"; python main.py
```

**Windows (CMD):**
```cmd
set PYTHONUTF8=1 && python main.py
```

**Linux / macOS:**
```bash
python main.py
```

> **Nota Windows/Unicode:** a variável PYTHONUTF8=1 é necessária em consoles Windows com
> codificação cp1252 (padrão em instalações em português). Sem ela, os caracteres de
> box-drawing usados nos logs (`─`, `═`, `●`) causam UnicodeEncodeError. Nenhum arquivo
> do projeto precisa ser alterado — apenas defina a variável antes de executar.

O script executa em sequência:
1. Carregamento e inspeção do dataset.
2. Análise exploratória (EDA) — gera 7 figuras em outputs/figures/.
3. Limpeza estrutural — gera data/processed/weatherAUS_processed.csv.
4. Treinamento, validação cruzada e tuning dos modelos (5 algoritmos).
5. Avaliação final no conjunto de teste — gera 5 figuras em outputs/figures/.
6. Exportação de métricas (4 tabelas CSV em outputs/tables/), top features e melhor modelo.

## Modelos implementados

| Modelo | Observação |
|---|---|
| DummyClassifier | Baseline majoritário |
| LogisticRegression | Baseline interpretável |
| DecisionTreeClassifier | Explicável, regras, importância |
| RandomForestClassifier | Ensemble robusto, importância |
| LinearSVC | SVM linear (amostra estratificada do treino por custo computacional) |

## Notas sobre reprodutibilidade

- random_state=42 em todos os pontos aplicáveis.
- Split treino/teste antes de qualquer ajuste de transformador.
- Imputação, encoding e scaling somente dentro de Pipeline/ColumnTransformer.
- O conjunto de teste é usado **apenas** na avaliação final.

## Resultados obtidos

Execução real com o dataset completo (142.193 registros, 5 modelos):

| Modelo | F1 | ROC-AUC | Recall | Precision | Accuracy |
|---|---|---|---|---|---|
| **RandomForest** | **0,6589** | **0,8868** | 0,6536 | 0,6643 | 0,8483 |
| LogReg | 0,6230 | 0,8667 | **0,7724** | 0,5221 | 0,7905 |
| DecTree | 0,5965 | 0,8472 | 0,7587 | 0,4914 | 0,7699 |
| SVM | 0,5853 | 0,8647 | 0,4904 | **0,7258** | 0,8442 |
| Baseline | 0,0000 | 0,5000 | 0,0000 | 0,0000 | 0,7758 |

**Melhor modelo: RandomForest** — F1 = 0,6589, ROC-AUC = 0,8868.  
Variáveis mais importantes: Humidity3pm (0,1478), Pressure3pm (0,0628).

## Documentação

- **Relatório final completo:** [docs/RELATORIO_FINAL.md](docs/RELATORIO_FINAL.md)
- **Estrutura do artigo científico:** [docs/artigo_estrutura.md](docs/artigo_estrutura.md)
- **Referências bibliográficas:** [docs/referencias.md](docs/referencias.md)
- **Figuras geradas:** outputs/figures/ (12 arquivos PNG)
- **Tabelas geradas:** outputs/tables/ (5 arquivos CSV)
