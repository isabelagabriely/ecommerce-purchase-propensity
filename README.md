# E-commerce Purchase Propensity

Projeto end-to-end de **Engenharia de Machine Learning** para previsão da propensão de compra de usuários de e-commerce com base em seu comportamento de navegação.

O projeto prioriza **reprodutibilidade, rastreabilidade de experimentos, versionamento de dados, qualidade de código e containerização**, utilizando Scikit-Learn, DVC, MLflow, Poetry, Docker, Pytest e Ruff.

## Problema de Negócio

Uma empresa de e-commerce deseja identificar sessões de navegação com maior probabilidade de resultar em uma compra.

Um modelo de propensão de compra pode apoiar estratégias como:

- campanhas direcionadas;
- incentivos personalizados;
- recomendações de produtos;
- priorização de sessões com maior intenção de compra.

A variável alvo é:

```text
Revenue
0 → a sessão não resultou em compra
1 → a sessão resultou em compra
```

## Dataset

O projeto utiliza o **Online Shoppers Purchasing Intention Dataset**, disponibilizado pelo UCI Machine Learning Repository.

O dataset contém:

```text
12.330 sessões de navegação
18 colunas
17 features de entrada
1 variável alvo binária: Revenue
```

O dataset bruto é baixado automaticamente pelo pipeline, portanto não é necessário realizar download manual.

Referência do dataset:

> Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository. DOI: 10.24432/C5F88Q.

## Arquitetura

```text
                    UCI Dataset
                        │
                        ▼
                  DVC: download
                        │
                        ▼
                   dados brutos
                        │
                        ▼
                DVC: preprocess
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          train.csv            test.csv
              │
              ▼
                   DVC: train
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
Logistic Regression              Random Forest
         │                             │
         └──────────────┬──────────────┘
                        ▼
             Stratified 5-fold CV
                        │
                        ▼
              Average Precision
                        │
                        ▼
                    Champion
                        │
                        ▼
               Held-out Test
                        │
                        ▼
                MLflow Tracking
                        │
                        ▼
             MLflow Model Registry
                        │
                        ▼
                   @champion
```

## Estrutura do Projeto

```text
ecommerce-purchase-propensity/
├── configs/
│   └── logging.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── model_comparison.csv
│   └── test_metrics.json
│
├── src/
│   └── ecommerce_purchase_propensity/
│       ├── __init__.py
│       ├── config.py
│       ├── logging_config.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── download.py
│       │   ├── preprocess.py
│       │   └── schema.py
│       │
│       └── modeling/
│           ├── __init__.py
│           ├── evaluate.py
│           ├── models.py
│           ├── registry.py
│           ├── tracking.py
│           └── train.py
│
├── tests/
│   ├── conftest.py
│   ├── test_evaluate.py
│   ├── test_logging.py
│   ├── test_models.py
│   ├── test_preprocessing.py
│   └── test_registry.py
│
├── .dockerignore
├── .dvcignore
├── .env.example
├── .gitignore
├── compose.yaml
├── Dockerfile
├── dvc.lock
├── dvc.yaml
├── params.yaml
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Requisitos

O projeto utiliza:

```text
Python 3.13
Poetry
Docker Desktop
```

As dependências Python são declaradas no `pyproject.toml` e suas versões resolvidas são registradas no `poetry.lock`.

As dependências de produção e desenvolvimento são mantidas separadamente.

## Configuração do Ambiente Local

Clone o repositório e acesse a pasta do projeto:

```bash
git clone https://github.com/isabelagabriely/ecommerce-purchase-propensity.git
cd ecommerce-purchase-propensity
```

Instale as dependências do projeto:

```bash
poetry install
```

Crie o arquivo local de variáveis de ambiente a partir do exemplo:

```bash
cp .env.example .env
```

A configuração local padrão é:

```text
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=ecommerce-purchase-propensity
MLFLOW_REGISTERED_MODEL_NAME=purchase-propensity-classifier
MLFLOW_MODEL_ALIAS=champion
LOG_LEVEL=INFO
```

O arquivo `.env` é propositalmente ignorado pelo Git.

## Pipeline DVC

O DVC é responsável pelo versionamento dos dados e pela reprodução do pipeline de Machine Learning.

O pipeline atual possui os seguintes estágios:

```text
download
   ↓
preprocess
   ↓
train
   ↓
register
```

Para visualizar o grafo do pipeline:

```bash
poetry run dvc dag
```

Para verificar se existem alterações em dependências, parâmetros ou outputs:

```bash
poetry run dvc status
```

Os parâmetros do pipeline são armazenados em:

```text
params.yaml
```

A configuração atual de dados é:

```yaml
data:
  test_size: 0.20
  random_state: 42
```

O dataset é obtido automaticamente do UCI Machine Learning Repository durante o estágio `download`.

Os arquivos de dados brutos e processados são tratados como outputs do DVC e não são versionados diretamente pelo Git.

## Aquisição dos Dados

A aquisição do dataset é automatizada pelo módulo:

```text
src/ecommerce_purchase_propensity/data/download.py
```

Também é possível executar apenas essa etapa diretamente:

```bash
poetry run python -m ecommerce_purchase_propensity.data.download
```

Para forçar um novo download:

```bash
poetry run python -m ecommerce_purchase_propensity.data.download --force
```

O processo valida:

```text
quantidade esperada de registros
schema esperado
presença das colunas necessárias
integridade básica do dataset
```

## Pré-processamento

O estágio de preprocessing realiza:

```text
validação de schema
        ↓
normalização do target
        ↓
split estratificado
        ↓
80% treino / 20% teste
```

A seed utilizada é:

```text
42
```

garantindo reprodutibilidade do split.

As transformações das features não são ajustadas antes da separação entre treino e teste.

Elas são incluídas diretamente dentro do `Pipeline` do Scikit-Learn, garantindo que os transformadores sejam ajustados apenas com dados de treinamento.

### Features Numéricas

As features numéricas utilizam:

```text
imputação pela mediana
        ↓
StandardScaler
```

### Features Categóricas

As features categóricas utilizam:

```text
imputação pelo valor mais frequente
        ↓
OneHotEncoder(handle_unknown="ignore")
```

Colunas como:

```text
Browser
Region
TrafficType
OperatingSystems
```

são tratadas como categóricas, mesmo sendo representadas originalmente por códigos numéricos.

## Treinamento dos Modelos

Dois modelos clássicos de classificação são comparados:

```text
Logistic Regression
Random Forest
```

A comparação utiliza **Stratified 5-Fold Cross-Validation** somente sobre os dados de treinamento.

A principal métrica utilizada para seleção do modelo campeão é:

```text
Average Precision
```

Essa métrica foi escolhida devido ao desbalanceamento entre sessões com e sem compra.

Também são acompanhadas as métricas:

```text
ROC-AUC
Accuracy
F1
Precision
Recall
```

O conjunto de teste permanece separado durante a seleção dos modelos e é utilizado somente após a definição do champion.

## Resultados dos Modelos

Resultados médios da validação cruzada:

| Modelo | Average Precision | ROC-AUC | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6457 | 0.8944 | 0.5144 | 0.7477 | 0.3925 |
| **Random Forest** | **0.7454** | **0.9263** | **0.6217** | **0.7665** | **0.5229** |

O **Random Forest** apresentou o maior Average Precision na validação cruzada e foi selecionado como modelo champion.

### Resultado no Conjunto de Teste

| Métrica | Resultado |
|---|---:|
| Average Precision | **0.7239** |
| ROC-AUC | **0.9168** |
| Accuracy | **0.8970** |
| F1 | **0.5890** |
| Precision | **0.7712** |
| Recall | **0.4764** |

O modelo apresenta comportamento relativamente conservador utilizando o threshold padrão de classificação.

A precision elevada indica que grande parte das sessões classificadas como compra realmente resulta em compra, enquanto o recall menor indica que parte dos compradores ainda não é identificada pelo modelo.

A otimização do threshold pode ser explorada futuramente de acordo com o custo de negócio associado a falsos positivos e falsos negativos.

Os resultados da comparação são armazenados em:

```text
reports/model_comparison.csv
```

As métricas finais do champion são armazenadas em:

```text
reports/test_metrics.json
```

## MLflow Tracking

O MLflow é utilizado para rastrear os experimentos de treinamento.

Cada modelo candidato gera uma run independente.

Para iniciar o MLflow Tracking Server localmente:

```bash
poetry run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

A interface do MLflow fica disponível em:

```text
http://127.0.0.1:5000
```

Cada run registra:

```text
hiperparâmetros do modelo
configuração da validação cruzada
métricas de validação
pipeline Scikit-Learn treinado
model signature
input example
params.yaml
dvc.lock
```

Para executar o treinamento rastreado pelo MLflow:

```bash
poetry run dvc repro train
```

Quando dados, parâmetros ou código upstream são alterados, o DVC determina quais estágios precisam ser reproduzidos.

Caso seja necessário criar intencionalmente um novo experimento de treinamento sem alterar uma dependência upstream:

```bash
poetry run dvc repro --force --single-item train
```

## MLflow Model Registry

Após a comparação dos modelos, a run vencedora é marcada como champion.

O modelo campeão é então registrado no **MLflow Model Registry**.

Para registrar o champion atual:

```bash
poetry run dvc repro --force --single-item register
```

A opção:

```text
--single-item
```

impede que o estágio de registro reproduza implicitamente o treinamento.

A opção:

```text
--force
```

é utilizada porque o registro altera um estado externo no MLflow Model Registry que não é representado apenas por outputs de arquivos do DVC.

O modelo registrado utiliza o nome:

```text
purchase-propensity-classifier
```

O champion atual pode ser acessado pelo alias:

```text
models:/purchase-propensity-classifier@champion
```

O uso do alias permite que consumidores do modelo não dependam de uma versão numérica específica.

Caso um modelo melhor seja registrado futuramente, o alias `champion` pode ser movido para a nova versão sem alterar a URI utilizada pelos consumidores.

## Execução Local Recomendada

Em um primeiro terminal, inicie o MLflow:

```bash
poetry run mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Em outro terminal, execute o pipeline até o treinamento:

```bash
poetry run dvc repro train
```

Depois registre o champion:

```bash
poetry run dvc repro --force --single-item register
```

Acesse os experimentos e o Model Registry em:

```text
http://127.0.0.1:5000
```

## Docker

O projeto pode ser executado em containers sem depender do ambiente Python instalado na máquina host.

O Docker Compose possui dois serviços:

```text
mlflow
   └── MLflow Tracking Server
       + Model Registry

pipeline
   └── DVC Pipeline
       ├── download
       ├── preprocess
       ├── train
       └── register
```

### Build

Construa as imagens:

```bash
docker compose build
```

### MLflow

Inicie o serviço MLflow:

```bash
docker compose up -d mlflow
```

A interface fica disponível em:

```text
http://localhost:5000
```

O estado do serviço pode ser verificado com:

```bash
docker compose ps
```

### Pipeline Completo

Execute o pipeline completo dentro do container:

```bash
docker compose run --rm pipeline
```

O container do pipeline se comunica com o MLflow através da rede interna do Docker:

```text
http://mlflow:5000
```

A execução percorre:

```text
download
   ↓
preprocess
   ↓
train
   ↓
register
```

O diretório `.git` é propositalmente excluído da imagem Docker.

Por isso, dentro da imagem, o DVC utiliza:

```text
core.no_scm=true
```

No ambiente de desenvolvimento local, a integração normal entre Git e DVC continua habilitada.

### Persistência do MLflow

O estado do MLflow é persistido através do volume Docker:

```text
mlflow-data
```

Esse volume mantém:

```text
experimentos
runs
métricas
model artifacts
Model Registry
```

mesmo após a recriação dos containers.

Para parar o ambiente sem remover os dados persistidos:

```bash
docker compose down
```

Para realizar um reset completo:

```bash
docker compose down -v
```

O parâmetro `-v` remove também o volume persistente e deve ser utilizado apenas quando um reset completo do ambiente for desejado.

> Cada execução completa do container `pipeline` representa uma nova execução do pipeline e pode gerar novas runs no MLflow e novas versões de modelos. Portanto, o pipeline completo não deve ser executado repetidamente sem intenção de criar um novo experimento.

## Testes

O projeto utiliza Pytest para validação automatizada.

A suíte atual cobre:

```text
configuração de logging
normalização do target
split estratificado e reproduzível
preprocessing
construção dos pipelines dos modelos
fit e predict dos modelos candidatos
seleção automática do champion
leitura e validação do relatório utilizado pelo Registry
```

Execute os testes com:

```bash
poetry run pytest
```

Resultado atual:

```text
7 passed
```

Para visualizar os testes individualmente:

```bash
poetry run pytest -v
```

## Qualidade de Código

O projeto utiliza Ruff para linting e formatação.

Execute o lint:

```bash
poetry run ruff check .
```

Valide a formatação:

```bash
poetry run ruff format --check .
```

Para corrigir automaticamente problemas suportados:

```bash
poetry run ruff check . --fix
```

Para aplicar formatação:

```bash
poetry run ruff format .
```

## Logging

O logging da aplicação é configurado através de:

```text
configs/logging.yaml
```

O nível de logging pode ser controlado através da variável:

```text
LOG_LEVEL
```

Exemplo:

```env
LOG_LEVEL=INFO
```

As principais etapas do pipeline registram eventos como:

```text
download do dataset
validação dos dados
preprocessing
split dos dados
treinamento
cross-validation
seleção do champion
criação das runs no MLflow
registro do modelo
atribuição do alias champion
```

## Reprodutibilidade

O projeto utiliza diferentes mecanismos para garantir reprodutibilidade:

```text
Poetry
poetry.lock
Python 3.13
seeds fixadas
split estratificado
Stratified K-Fold
params.yaml
DVC
dvc.lock
download automatizado do dataset
MLflow Tracking
MLflow Model Registry
Docker
Docker Compose
```

## Stack Tecnológica

| Área | Tecnologia |
|---|---|
| Linguagem | Python 3.13 |
| Gerenciamento de dependências | Poetry |
| Machine Learning | Scikit-Learn |
| Manipulação de dados | Pandas / NumPy |
| Versionamento de dados | DVC |
| Pipeline reprodutível | DVC |
| Tracking de experimentos | MLflow |
| Model Registry | MLflow Model Registry |
| Containerização | Docker / Docker Compose |
| Testes | Pytest |
| Linting e formatação | Ruff |
| Configuração | YAML / `.env` |
| Logging | Python Logging |

## Limitações e Melhorias Futuras

O projeto prioriza Engenharia de Machine Learning e reprodutibilidade, não otimização extensiva do modelo.

Possíveis evoluções incluem:

```text
otimização do threshold de classificação
novos modelos candidatos
otimização de hiperparâmetros
feature engineering
calibração de probabilidades
DVC Remote
backend remoto para MLflow
CI/CD
deploy em cloud
monitoramento de modelo
detecção de drift
```

Essas evoluções podem ser adicionadas posteriormente sem alterar a arquitetura principal do pipeline.
