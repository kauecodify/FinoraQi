# FinoraQi v2

A versão 2 da FinoraQi reúne em um só projeto a interface, a API, a inteligência artificial, o machine learning, o processamento de dados e a nova camada CashPool B2B. Ela fica nesta pasta `v2/` e não depende do código da versão 1: o site e os backends originais continuam intactos na raiz do repositório.

## Sumário

1. [Visão geral](#visão-geral)
2. [Diferenças entre a versão 1 e a versão 2](#diferenças-entre-a-versão-1-e-a-versão-2)
3. [Estrutura de pastas](#estrutura-de-pastas)
4. [Como executar](#como-executar)
5. [Testes e qualidade](#testes-e-qualidade)
6. [API](#api)
7. [Inteligência artificial e machine learning](#inteligência-artificial-e-machine-learning)
8. [Processamento de dados](#processamento-de-dados)
9. [CashPool B2B](#cashpool-b2b)
10. [Configuração](#configuração)
11. [Limitações conhecidas e próximos passos](#limitações-conhecidas-e-próximos-passos)

## Visão geral

| Camada | Tecnologia |
| --- | --- |
| Frontend | HTML, CSS e JavaScript sem framework e sem etapa de build (`frontend/index.html` e `frontend/app.html`) |
| API | FastAPI, uma única aplicação com rotas versionadas em `/api/v1` |
| Persistência | SQLAlchemy 2, Alembic, SQLite em desenvolvimento e PostgreSQL em produção |
| Autenticação | JWT (PyJWT) com senha em bcrypt e isolamento por tenant |
| Machine learning | NumPy, Numba, scikit-learn, pandas, joblib |
| Inteligência artificial | Categorização de transações por texto, insights financeiros e previsão de saldo |
| Processamento de dados | Leitura, limpeza, perfil, agregação e exportação de CSV, XLSX e XLS |
| B2B | CashPool: sweep, netting, câmbio, rendimento, previsão de liquidez e compliance |

A mesma aplicação serve a API e o frontend. Com o servidor no ar, a landing page fica em `/`, a plataforma em `/app.html` e a documentação interativa da API em `/docs`.

## Diferenças entre a versão 1 e a versão 2

### Resumo

| Tema | Versão 1 | Versão 2 |
| --- | --- | --- |
| Organização | Três backends em paralelo (FastAPI em `backend-fastapi/`, Express em `js/server.js`, scripts em `ML_V1/` e `quantics/`) e páginas HTML soltas em `public/` | Um único projeto em `v2/` com backend, frontend, testes, migrações e infraestrutura |
| Aplicação FastAPI | Duas instâncias de `FastAPI` no mesmo `main.py` e a rota WebSocket `/ws/payment/{user_id}` declarada duas vezes | Uma fábrica `create_app()`, um ciclo de vida (`lifespan`) e um único WebSocket autenticado |
| Imports | Referências a módulos inexistentes (`app.api.v1.payment`, `app.db.database`, `app.core.websocket_manager`, `app.ml.automl_engine`) impediam a API de iniciar | Todos os módulos existem; o import da aplicação e os testes rodam sem erro |
| Dependências | `requirements.txt` com marcadores de conflito do Git; `Dockerfile` com vários Dockerfiles concatenados; `docker-compose.yml` com volumes inválidos | `requirements.txt` com versões fixas, `Dockerfile` multi-stage com usuário sem privilégios e healthcheck, `docker-compose.yml` validado com PostgreSQL |
| Banco de dados | SQLite no Express e PostgreSQL no FastAPI, sem migrações e com esquemas divergentes | Um modelo SQLAlchemy único, migração Alembic `0001` e o mesmo esquema em SQLite e PostgreSQL |
| Segurança | Segredos JWT fixos no código (`finoraqi_secret_key_2026`, `dev_secret_change_in_prod`) | Segredo via variável de ambiente com no mínimo 32 caracteres, bcrypt, expiração de token, validação de entrada com Pydantic e isolamento por tenant em todas as consultas |
| Score de risco de pagamento | `calculate_risk_score` exigia 14 argumentos, mas era chamado com 4; a decisão estava invertida (score alto era aprovado) | Convenção explícita: 0 é risco mínimo e 1 é risco máximo. Baixo risco é aprovado, médio ou anômalo vai para revisão, alto é rejeitado. NumPy e Numba produzem o mesmo resultado (verificado em teste) |
| Aprendizado incremental | `LabelEncoder` reajustado a cada amostra e `StandardScaler` treinado com uma única linha | Codificador de categorias estável, `partial_fit` em mini-lotes, classes declaradas no primeiro lote e categorias novas tratadas |
| AutoML | Dependia de Celery e Redis, com imports incompatíveis entre `tasks.py` e `tasky.py` | Pipeline scikit-learn com imputação, padronização e one-hot, comparação de modelos por validação cruzada, holdout, persistência por tenant e execução em segundo plano sem infraestrutura extra |
| Otimização de hiperparâmetros | Módulo `quantics` dependia de uma biblioteca não confirmada e de imports quebrados | Busca evolutiva tipada (`app/ml/evolution.py`) que preserva inteiros e limites de cada parâmetro |
| Score de crédito | Regra fixa sem modelo estatístico | Regressão logística que estima a probabilidade de inadimplência e converte para a escala 300 a 900, com validação de CPF |
| Inteligência artificial | Não havia categorização automática nem recomendações | Categorizador TF-IDF com Naive Bayes que aprende com correções do usuário, insights financeiros e previsão de saldo com faixas p10, p50 e p90 |
| Processamento de dados | Leitura de planilhas espalhada e sem tratamento de formatos brasileiros | Módulo único que normaliza colunas, entende `R$ 1.234,56` e datas `dd/mm/aaaa`, remove duplicatas, preenche ausentes, gera perfil, agrega e exporta, com limite de linhas e de tamanho |
| CashPool B2B | Não existia | Módulo completo com sweep, netting multilateral, câmbio, rendimento por TWAB, previsão de liquidez, compliance por país e feature flags |
| Frontend | Várias páginas com estilos próprios, partículas animadas, emojis, Font Awesome e um cliente de API com marcadores de conflito do Git | Landing minimalista com a mesma paleta e uma plataforma de página única que consome a API real, sem dependências externas além das fontes |
| Tempo real | Duas rotas WebSocket idênticas e sem autenticação | Um WebSocket por cliente, autenticado por token e isolado por tenant |
| Testes | Nenhuma suíte automatizada executável | 42 testes em pytest cobrindo API, ML, IA, dados e CashPool, além de lint e formatação com Ruff |

### Interface

A paleta foi mantida integralmente: ciano `#00d4ff`, roxo `#7b2cbf`, violeta `#b700ff`, dourado `#ffd700` e `#ffb700`, azul-marinho `#1a1a2e`, verde `#10b981`, vermelho `#ef4444` e texto `#e2e8f0`. O que mudou foi o uso das cores:

- O azul-marinho virou o fundo e o gradiente ciano para roxo aparece só como destaque.
- O dourado ficou reservado para as ações principais, o verde para estados positivos e o vermelho para risco.
- Saíram as partículas animadas, o logo girando, os emojis, o letreiro de cotações com valores fixos e a dependência do Font Awesome. Os ícones são SVG embutidos.
- A tipografia passou a usar Inter para texto e JetBrains Mono para números e código.
- A animação respeita a preferência do sistema por movimento reduzido.

A landing (`frontend/index.html`) apresenta o CashPool como quarto produto e traz um exemplo interativo de netting. Os números desse exemplo (quatro obrigações somando 350 mil viram duas transferências somando 20 mil, economia de 94%) são reproduzidos pelo motor real e verificados no teste `test_landing_page_netting_example_matches_engine`.

A plataforma (`frontend/app.html`) substitui as páginas `dashboard.html`, `calculadora.html`, `automl_dashboard.html`, `crm_enterprise.html` e `onboarding_core.html` da versão 1 por sete telas:

| Tela | Conteúdo |
| --- | --- |
| Visão geral | Receitas, despesas, saldo, taxa de poupança, insights da IA, despesas por categoria e projeção de saldo |
| Transações | Cadastro com categorização automática e histórico |
| Pagamentos e risco | Simulação de pagamento, score de risco, motivos da decisão, eventos em tempo real e benchmark Numba contra NumPy |
| Score de crédito | Registro de operações e consulta por CPF |
| Dados e AutoML | Envio de CSV ou XLSX, análise, perfil, treino, comparação de modelos e download do modelo |
| CashPool B2B | Grupo demo, posições consolidadas, sweep, netting, rendimento e histórico |
| Feature flags | Ativação por tenant, restrita a administradores |

### Compatibilidade com a versão 1

- As rotas principais continuam sob `/api/v1` e mantêm os nomes da versão 1 (`/auth/register`, `/auth/login`, `/profile`, `/transactions`, `/investments`, `/goals`, `/wallets`, `/payments/process`, `/credit-operation`, `/score/{cpf}`), com os campos `nome`, `email` e `senha` no cadastro.
- Mudanças de contrato: a resposta de autenticação passou a ser `{"token", "token_type", "user"}`; o WebSocket mudou de `/ws/payment/{user_id}` para `/api/v1/ws/payments/{user_id}?token=...`; o score de risco agora significa risco (quanto maior, pior).
- Os bancos da versão 1 não são migrados automaticamente. Para importar dados antigos, exporte as transações em CSV e use `POST /api/v1/data/clean` ou cadastre-as via `POST /api/v1/transactions`.

## Estrutura de pastas

```
v2/
  README.md
  .env.example
  docker-compose.yml
  backend/
    Dockerfile
    requirements.txt
    requirements-dev.txt
    pyproject.toml              configuração do Ruff e do pytest
    alembic.ini
    alembic/versions/           migração inicial do esquema
    app/
      main.py                   fábrica da aplicação e ciclo de vida
      core/                     configuração, segurança, dependências, feature flags, WebSocket, jobs, cache
      db/                       base, sessão e modelos
      api/v1/                   auth, finance, payments, score, flags, data, ai, automl
      ml/                       scoring, anomaly, credit_risk, incremental, automl_engine, forecast, evolution
      ai/                       categorizer, insights
      processing/               spreadsheet
      modules/cashpool/         modelos, motores, serviço e rotas do CashPool
    tests/                      suíte pytest
  frontend/
    index.html                  landing page
    app.html                    plataforma
    assets/api.js               cliente da API
    assets/img/                 logo e marcas parceiras
```

## Como executar

### Local

Requer Python 3.10 ou superior.

```bash
cd v2/backend
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp ../.env.example .env          # ajuste JWT_SECRET_KEY
uvicorn app.main:app --reload --port 8000
```

Acesse `http://localhost:8000` (landing), `http://localhost:8000/app.html` (plataforma) e `http://localhost:8000/docs` (API). Em desenvolvimento as tabelas são criadas automaticamente no SQLite. Para usar migrações, defina `AUTO_CREATE_TABLES=false` e rode `alembic upgrade head`.

O primeiro usuário cadastrado em um banco novo recebe o papel `admin`.

### Docker

```bash
cd v2
cp .env.example .env             # ajuste JWT_SECRET_KEY e POSTGRES_PASSWORD
docker compose up -d --build
```

O compose sobe o PostgreSQL 16 e a API, aplica as migrações Alembic na inicialização e publica a aplicação na porta 8000 (altere com `API_PORT`). Se o Docker Hub limitar downloads, use um espelho com `PYTHON_IMAGE=mirror.gcr.io/library/python:3.11-slim` e `POSTGRES_IMAGE=mirror.gcr.io/library/postgres:16-alpine`.

## Testes e qualidade

```bash
cd v2/backend
pytest -q
ruff check .
ruff format --check .
```

| Arquivo | Cobertura |
| --- | --- |
| `test_auth_finance.py` | Cadastro, login, perfil, token obrigatório, CRUD financeiro, resumo e isolamento entre usuários |
| `test_payments_score.py` | Decisão de risco, anomalia, lote, benchmark, validação, WebSocket e score de crédito |
| `test_ml_units.py` | Equivalência NumPy e Numba, monotonicidade do risco, anomalia, crédito, previsão, AutoML, aprendizado incremental e busca evolutiva |
| `test_data_ai.py` | Limpeza de CSV brasileiro, endpoints de dados, categorização, feedback, insights e previsão |
| `test_automl_api.py` | Análise, treino, predição, listagem, download, jobs em segundo plano, incremental e isolamento de modelos por tenant |
| `test_cashpool.py` | Netting, compliance, câmbio, validações, sweep, rendimento, previsão de liquidez, isolamento por tenant, feature flags e pool nocional |

## API

Todas as rotas abaixo ficam sob `/api/v1` e, exceto cadastro e login, exigem `Authorization: Bearer <token>`. A referência completa com exemplos está em `/docs`.

| Grupo | Rotas |
| --- | --- |
| Saúde | `GET /health` e `GET /api/v1/health` |
| Autenticação | `POST /auth/register`, `POST /auth/login`, `GET /profile`, `PUT /profile` |
| Finanças | `GET POST /transactions`, `PUT DELETE /transactions/{id}`, o mesmo para `/investments`, `/goals` e `/wallets`, `GET /dashboard/summary` |
| Pagamentos | `POST /payments/process`, `POST /payments/batch`, `GET /payments`, `GET /payments/benchmark`, `WS /ws/payments/{user_id}?token=` |
| Crédito | `POST /credit-operation`, `GET /score/{cpf}` |
| IA | `GET /ai/categories`, `POST /ai/categorize`, `POST /ai/categorize/feedback`, `GET /ai/insights`, `POST /ai/forecast` |
| Dados | `POST /data/profile`, `POST /data/clean`, `POST /data/aggregate`, `POST /data/export` |
| AutoML | `POST /automl/analyze`, `POST /automl/train`, `POST /automl/evolve`, `POST /automl/incremental`, `GET /automl/jobs/{id}`, `GET /automl/models`, `POST /automl/predict`, `GET /automl/models/{id}/download` |
| Feature flags | `GET /flags`, `PUT /flags/{chave}` (somente admin) |
| CashPool | ver a seção [CashPool B2B](#cashpool-b2b) |

Exemplo de pagamento:

```bash
curl -X POST http://localhost:8000/api/v1/payments/process \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"user_id": "cliente_1", "amount": "250.00", "description": "Compra", "payment_method": "pix"}'
```

```json
{"transaction_id": "txn_...", "status": "approved", "risk_score": 0.06, "risk_level": "low", "is_anomaly": false, "reasons": ["..."]}
```

## Inteligência artificial e machine learning

| Módulo | O que faz |
| --- | --- |
| `ml/scoring.py` | Score de risco de pagamento com oito atributos (valor, horário, fim de semana, método, categoria, frequência, desvio em relação ao histórico e confiança do cliente). Implementação vetorizada em NumPy e compilada em Numba, com fallback automático quando o Numba não está disponível. Limiares: abaixo de 0,35 baixo, a partir de 0,70 alto |
| `ml/anomaly.py` | Z-score robusto (mediana e MAD) sobre o histórico do cliente e regra para criptoativos acima de R$ 10.000 |
| `ml/credit_risk.py` | Regressão logística determinística para probabilidade de inadimplência, convertida para 300 a 900 com faixas de baixo a excelente |
| `ml/incremental.py` | Treino online com `SGDClassifier` ou `SGDRegressor`, `StandardScaler.partial_fit` e codificação estável de categorias |
| `ml/automl_engine.py` | Detecta classificação ou regressão, monta o pré-processamento, compara Regressão Logística, Random Forest e Gradient Boosting (ou Ridge, Random Forest e Gradient Boosting na regressão) por validação cruzada e avalia o melhor em holdout |
| `ml/evolution.py` | Busca evolutiva de hiperparâmetros com seleção, cruzamento e mutação tipados |
| `ml/forecast.py` | Modelo de fluxo de caixa com tendência e sazonalidade semanal por mínimos quadrados, com faixas p10, p50 e p90 a partir dos resíduos |
| `ai/categorizer.py` | Categorização de descrições em português com TF-IDF de caracteres e Naive Bayes; aprende com o feedback do usuário |
| `ai/insights.py` | Agregações com pandas, recomendações (gastos acima da renda, poupança baixa, concentração por categoria, despesas atípicas, risco de saldo negativo) e previsão de saldo |

Os modelos treinados pelo AutoML são gravados com joblib em `MODELS_DIR/<tenant>/` e registrados na tabela `ml_models`. Um tenant não consegue listar, usar ou baixar modelos de outro.

## Processamento de dados

`app/processing/spreadsheet.py` concentra o tratamento de planilhas:

- Aceita CSV (com detecção de separador e leitura em UTF-8 ou Latin-1), XLSX e XLS, com limite de tamanho (`MAX_UPLOAD_MB`) e de linhas (200.000 por padrão).
- Normaliza nomes de colunas para `snake_case` sem acentos.
- Converte valores monetários brasileiros (`R$ 1.234,56`) e internacionais e datas (`dd/mm/aaaa` e ISO).
- Remove linhas e colunas vazias e duplicatas, e preenche ausentes numéricos por mediana, média ou zero (ou mantém sem preenchimento com `none`).
- Gera perfil por coluna (tipo, ausentes, únicos, estatísticas e outliers por IQR) e um relatório do que foi alterado.
- Agrega com uma lista fechada de funções (`sum`, `mean`, `median`, `min`, `max`, `count`) e exporta em CSV ou XLSX.

## CashPool B2B

O CashPool consolida o caixa de uma holding e de suas subsidiárias em uma conta header. Todo valor monetário usa `Decimal` com arredondamento bancário explícito.

| Motor | Arquivo | Função |
| --- | --- | --- |
| Sweep | `sweep.py` | Leva o saldo de cada membro ao alvo (zero, valor fixo ou previsão), respeitando saldo mínimo e máximo, direção permitida e valor mínimo de sweep. Suporta simulação |
| Netting | `netting.py` | Compensa obrigações intercompany de forma bilateral ou multilateral e registra quem paga quem, o volume bruto, o líquido e a economia |
| Câmbio | `fx.py` | Provedor estático determinístico ou HTTP (taxas do BCE via frankfurter.app) com cache e fallback |
| Rendimento | `yield_alloc.py` | Distribui o rendimento do pool pelo saldo médio ponderado no tempo (TWAB), nos métodos pro rata, por faixas ou contratual, com soma exata ao centavo |
| Previsão | `forecast.py` | Projeta o saldo de cada membro a partir do histórico de posições |
| Compliance | `compliance.py` | Regras por país (BR, MX, EU, US, IN). No Brasil, sinaliza IOF e documentação de preço de transferência |
| Liquidação | `rails.py` | Livro-razão interno que registra débito e crédito como novas posições |

Rotas, todas sob `/api/v1/cashpool` e controladas pela flag `cashpool.enabled`:

| Método | Rota | Flag adicional |
| --- | --- | --- |
| POST, GET | `/pools` | |
| GET | `/pools/{id}` | |
| POST, GET | `/pools/{id}/positions` | |
| POST | `/pools/{id}/sweep?dry_run=` | `cashpool.sweep` |
| POST | `/pools/{id}/obligations` | |
| POST | `/pools/{id}/netting` | `cashpool.netting` |
| POST | `/pools/{id}/yield` | `cashpool.yield` |
| GET | `/pools/{id}/forecast/{member_id}` | |
| GET | `/pools/{id}/history` | |

Correções feitas em relação ao rascunho do CashPool usado como referência:

- O controle por feature flag usa dependências do FastAPI, em vez de um decorator que não funcionava com as rotas.
- O tenant é tratado como identificador textual em todas as consultas.
- O motor de compliance usa regras concretas, sem instanciar classe abstrata, e exige pelo menos dois membros e exatamente um header.
- O TWAB é calculado a partir das posições reais; antes retornava zero e impedia qualquer alocação.
- O netting registra as liquidações resultantes, em vez de apenas zerar os saldos.
- A previsão de liquidez usa o modelo de fluxo de caixa existente no projeto.
- O câmbio funciona sem rede por padrão.
- Não há mistura de `float` com `Decimal` em cálculos monetários.
- Pools nocionais não executam sweep físico.

## Configuração

Variáveis lidas de `.env` (veja `.env.example`):

| Variável | Padrão | Descrição |
| --- | --- | --- |
| `ENV` | `development` | Ambiente |
| `DATABASE_URL` | `sqlite:///./finoraqi.db` | URL SQLAlchemy; em produção `postgresql+psycopg://...` |
| `AUTO_CREATE_TABLES` | `true` | Cria tabelas na inicialização; use `false` com Alembic |
| `JWT_SECRET_KEY` | valor de desenvolvimento | Obrigatório trocar em produção, no mínimo 32 caracteres |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `1440` | Validade do token |
| `CORS_ORIGINS` | localhost | Lista JSON de origens permitidas |
| `MODELS_DIR` | `./models_store` | Pasta dos modelos treinados |
| `FRONTEND_DIR` | `../frontend` | Pasta servida como site estático |
| `MAX_UPLOAD_MB` | `20` | Tamanho máximo de upload |
| `FX_PROVIDER` | `static` | `static` ou `http` |

## Limitações conhecidas e próximos passos

- Os jobs de AutoML rodam em um pool de threads do próprio processo. Com vários workers ou servidores, troque por uma fila dedicada (Celery, RQ ou Arq) mantendo a mesma interface de `app/core/jobs.py`.
- O cache e o gerenciador de WebSocket ficam em memória. Para escalar horizontalmente, use Redis para cache e pub/sub.
- O CashPool liquida em um livro-razão interno. A integração com bancos reais (Open Finance, SWIFT, PIX) deve implementar a interface `PaymentRail`.
- As taxas de câmbio estáticas são de referência. Em produção use `FX_PROVIDER=http` ou um provedor contratado.
- Os modelos de risco e crédito partem de dados sintéticos determinísticos. Antes de decisões reais, devem ser recalibrados com dados históricos da operação.
- O preço "Sob consulta" do CashPool na landing é provisório.
