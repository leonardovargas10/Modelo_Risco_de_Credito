---

## Análise de Risco de Crédito direcionada por Modelagem Matemática e Aprendizado de Máquina

<p align="center">
  <img src = './img01.jpg' width = '50%'>
</p>

> **Autor:** Leonardo Aderaldo Vargas  
> **Instituição:** UNESP Sorocaba — Bacharelado em Engenharia de Controle e Automação  
> **Natureza:** Trabalho de Conclusão de Curso (TCC)  
> **Fonte dos Dados:** [Kaggle — Lending Club Loan Data](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data)  
> **Status:**

<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=FINALIZADO&color=GREEN&style=for-the-badge"/>
</p>

---

## Sumário

1. [Contexto e Justificativa](#1-contexto-e-justificativa)
2. [Objetivos](#2-objetivos)
3. [Sobre o Lending Club](#3-sobre-o-lending-club)
4. [Fundamentação Teórica](#4-stack-tecnológica)
5. [Fontes de Dados e Dicionário de Variáveis](#5-fontes-de-dados-e-dicionário-de-variáveis)
6. [Arquitetura da Solução](#6-arquitetura-da-solução)
7. [Definição da Target — Probabilidade de Default (PD)](#7-definição-da-target--probabilidade-de-default-pd)
8. [Estratégia de Amostragem](#8-estratégia-de-amostragem)
9. [Análise Exploratória — Variáveis de Produto](#9-análise-exploratória--variáveis-de-produto)
10. [Análise Exploratória — Variáveis de Cliente](#10-análise-exploratória--variáveis-de-cliente)
11. [Feature Engineering](#11-feature-engineering)
12. [Feature Selection](#12-feature-selection)
13. [Modelagem Inicial — Comparativo de Algoritmos](#13-modelagem-inicial--comparativo-de-algoritmos)
14. [Otimização do Modelo Final](#14-otimização-do-modelo-final)
15. [Política de Crédito (Baseline Rules-Based)](#15-política-de-crédito-baseline-rules-based)
16. [Resultados Consolidados](#16-resultados-consolidados)
17. [Artefatos Gerados](#17-artefatos-gerados)
18. [Referências Bibliográficas](#18-referências-bibliográficas)

---

## 1. Contexto e Justificativa

O trabalho apresenta uma aplicação prática dos fundamentos de **Risco de Crédito** aliada a técnicas de **Matemática Aplicada, Estatística e Machine Learning**, com o objetivo de classificar clientes inadimplentes e subsidiar decisões de concessão de crédito de forma rápida, automática e matematicamente fundamentada.

A concessão de crédito é um dos processos mais críticos de uma instituição financeira. Erros nessa decisão — seja pela aprovação de crédito a maus pagadores ou pela recusa a bons clientes — geram impactos financeiros diretos. A solução proposta utiliza exclusivamente critérios quantitativos para **maximizar a confiança do credor e minimizar o risco de inadimplência**, substituindo análises subjetivas por um modelo preditivo robusto.

A base de dados, proveniente do Kaggle, é fictícia mas representa com fidelidade o ambiente real de uma instituição financeira, preservando a privacidade dos dados sensíveis.

---

## 2. Objetivos

Desenvolver um **software de análise de risco de crédito** capaz de:

- Classificar clientes em **bons pagadores** (GOOD) e **maus pagadores** (BAD) com base em dados cadastrais, comportamentais e de produto;
- Estimar a **Probabilidade de Default (PD)** individual de cada solicitante;
- Definir um **threshold de decisão** que maximize o retorno financeiro da instituição;
- Estabelecer uma **Política de Crédito** (baseline rules-based) utilizando técnicas de Information Value (IV) e os 5 C's do Crédito;
- Comparar a abordagem direcionada por regras vs abordagens matemáticas e discutir suas nuances
- Servir como referência técnica e acadêmica de aplicação de Machine Learning no mercado de crédito.

---

## 3. Sobre o Lending Club

O Lending Club é uma empresa norte-americana de empréstimos _peer-to-peer_, que conecta investidores a tomadores de crédito. Os investidores aportam capital que é repassado aos mutuários, e o retorno — capital mais juros — é devolvido aos investidores ao longo do prazo do empréstimo.

O conjunto de dados abrange empréstimos emitidos entre **2007 e 2015**, incluindo situação atual do contrato (adimplente, atrasado, quitado, cancelado etc.), pontuação de crédito, consultas financeiras, localização geográfica, histórico de inadimplência, entre outros atributos.

---

## 4. Fundamentação Teórica

- [x] Fundamentos de Risco de Crédito
- [x] Fundamentos de Modelagem de Crédito
- [x] Fundamentos de GIT
- [x] SQL
- [x] Python
- [x] Análise de Dados
- [x] Técnicas de Modelagem Matemática e Estatística
- [x] Técnicas de Machine Learning

---

## 5. Fontes de Dados e Dicionário de Variáveis

A base integra **variáveis de produto** (características do empréstimo contratado) e **variáveis de cliente** (características cadastrais e comportamentais do solicitante).

### Variável Target

| Campo | Descrição |
|---|---|
| `situacao_do_emprestimo` | Situação atual do contrato → classificado como **GOOD** (0) ou **BAD** (1) |

### Variáveis de Produto

| Campo | Descrição |
|---|---|
| `qt_parcelas` | Número de parcelas do empréstimo (36 ou 60 meses) |
| `grau_de_emprestimo` | Grau de risco do empréstimo atribuído pela plataforma (A–G) |
| `subclasse_de_emprestimo` | Subclasse do grau de empréstimo (A1–G5) |
| `produto_de_credito` | Finalidade declarada do empréstimo (consolidação de dívidas, educação, etc.) |
| `valor_emprestimo_solicitado` | Valor total do empréstimo solicitado pelo mutuário |
| `taxa_de_juros` | Taxa de juros aplicada ao contrato |
| `data_financiamento_emprestimo` | Data de emissão do empréstimo |
| `produto_disponivel_publicamente` | Flag de disponibilidade pública do produto |
| `plano_de_pagamento` | Flag de plano de pagamento especial implementado |
| `tipo_de_concessao_do_credor` | Status da listagem inicial (W ou F) |
| `pagamento_mensal` | Valor da parcela mensal devida |

### Variáveis de Cliente

| Campo | Descrição |
|---|---|
| `cargo_cliente` | Cargo/profissão declarado pelo solicitante |
| `qt_anos_mesmo_emprego` | Tempo no emprego atual (0 = < 1 ano; 10 = 10+ anos) |
| `status_propriedade_residencial` | Tipo de moradia (Aluguel, Própria, Hipoteca, Outros) |
| `renda_comprovada` | Flag de comprovação de renda (Verificada / Não Verificada) |
| `faturamento_anual` | Renda anual declarada pelo cliente |
| `estado` | Estado de residência do cliente |
| `limite_total_produtos_credito` | Limite total em todos os produtos de crédito vigentes |
| `limite_total_rotativos` | Limite total de crédito rotativo (cartão + cheque especial) |
| `limite_rotativos_utilizado` | Valor utilizado do crédito rotativo |
| `taxa_utilizacao_limite_rotativos` | Proporção utilizada do limite rotativo |
| `qt_produtos_credito_contratados_atualmente` | Produtos de crédito ativos no momento |
| `qt_produtos_credito_contratados_historicamente` | Total histórico de produtos contratados |
| `qt_registros_publicos_depreciativos` | Número de registros públicos negativos |
| `qt_consultas_credito_6meses` | Consultas de crédito nos últimos 6 meses |
| `data_contratacao_primeiro_produto_credito` | Data do primeiro produto de crédito contratado |
| `qt_meses_desde_ultimo_registro_publico` | Meses desde o último registro público |
| `qt_meses_classificacao_mais_recente_90dias` | Meses desde classificação de risco mais recente ≥ 90 dias |
| `qt_meses_ultima_inadimplencia` | Meses desde a última ocorrência de inadimplência |
| `qt_incidencias_inadimplencia_vencidas_30dias` | Incidências de inadimplência vencidas > 30 dias nos últimos 2 anos |

---

## 6. Arquitetura da Solução

```
Dados Brutos (Lending Club Loan Data — Kaggle)
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │         Definição da Target (PD)            │
 │  GOOD (0) vs BAD (1) — 5 critérios de       │
 │  inadimplência                              │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │   Separação Treino / Teste (80% / 20%)      │
 │   Estratificada — antes de qualquer EDA     │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────────┬──────────────────┐
 │  EDA — Variáveis         │  EDA — Variáveis │
 │  de Produto (NB 01)      │  de Cliente      │
 │  Chi-Quadrado + WOE      │  (NB 02)         │
 └──────────────────────────┴──────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │            Feature Engineering              │
 │  Anos no emprego, Comprometimento de renda, │
 │  Meses desde datas, Flags binárias,         │
 │  Target Encoder (Bad Rate)                  │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │             Feature Selection               │
 │  Variância → Informação Mútua →             │
 │  Random Forest Feature Importance           │
 └─────────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────────┬──────────────────────┐
 │  Modelagem Preditiva │  Política de Crédito │
 │  5 algoritmos        │  (Regras + IV +       │
 │  XGBoost Otimizado   │   5 C's do Crédito)  │
 └──────────────────────┴──────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────┐
 │   Threshold de Decisão + Retorno Financeiro │
 └─────────────────────────────────────────────┘
```

---

## 7. Definição da Target — Probabilidade de Default (PD)

Um cliente é classificado como **BAD (1 — Mau Pagador)** se atender a ao menos um dos seguintes critérios:

| Critério | Descrição |
|---|---|
| `Charged Off` | Empréstimo cancelado por inadimplência prolongada |
| `Default` | Status explícito de inadimplência |
| `Late (31-120 days)` | Atraso superior a 30 dias |
| `Does not meet credit policy. Status: Charged Off` | Não atende aos Critérios Mínimos de Aprovação |
| `qt_contas_inadimplentes_outras_instituicoes >= 1` | Possui conta em inadimplência em outra instituição |

Clientes que não se enquadram em nenhum desses critérios são classificados como **GOOD (0 — Bom Pagador)**.

```
Distribuição da base:
  GOOD (0) → ~88,74% dos clientes
  BAD  (1) → ~11,26% dos clientes
```

> O **desbalanceamento de classes** (~1:8) é tratado via `class_weight` durante a modelagem.

---

## 8. Estratégia de Amostragem

A separação treino/teste é realizada **antes de qualquer análise exploratória ou transformação**, garantindo total ausência de data leakage:

| Conjunto | Proporção | Uso |
|---|---|---|
| **Treino** | 80% | EDA, Feature Engineering, Feature Selection, treinamento dos modelos e ajuste dos encoders |
| **Teste** | 20% | Avaliação final do modelo — dados nunca vistos durante o desenvolvimento |

A separação é estratificada pela target, preservando a proporção de bons e maus pagadores em ambos os conjuntos.

---

## 9. Análise Exploratória — Variáveis de Produto

A EDA de variáveis de produto foi conduzida exclusivamente sobre o conjunto de **treino**, utilizando as seguintes ferramentas analíticas:

**Variáveis Categóricas:** Teste Chi-Quadrado (H₀: sem associação; H₁: há associação, α = 0,05) + Weight of Evidence (WOE).

**Variáveis Contínuas:** Boxplot, análise por decis e distribuição de inadimplentes por faixa de valor.

### Principais Descobertas

| Variável | Observação |
|---|---|
| `qt_parcelas` | Clientes de 60 parcelas apresentam maior PD (WOE = +0,30 vs −0,13 em 36 parcelas) |
| `grau_de_emprestimo` | Forte associação com inadimplência — graus mais altos (F, G) têm maior risco |
| `subclasse_de_emprestimo` | Idem ao grau — subclasses de menor rating apresentam maior PD |
| `produto_de_credito` | Produtos como `small_business`, `medical` e `moving` têm maior risco; `debt_consolidation` e `credit_card` têm menor risco |
| `valor_emprestimo_solicitado` | Pouca discriminação isolada — tanto valores altos quanto baixos apresentam risco |
| `taxa_de_juros` | **Forte discriminação** — taxas mais altas correlacionam-se com maior PD (efeito produto) |
| `data_financiamento_emprestimo` | Empréstimos mais antigos (mais meses desde o financiamento) apresentam maior risco |
| `produto_disponivel_publicamente` | Sem poder discriminativo |
| `plano_de_pagamento` | Nenhum cliente da base aderiu a plano especial |
| `tipo_de_concessao_do_credor` | Listagem "w" associada a menor risco que "f" |

---

## 10. Análise Exploratória — Variáveis de Cliente

### Principais Descobertas

| Variável | Observação |
|---|---|
| `qt_anos_mesmo_emprego` | Clientes com < 3 anos no emprego têm maior PD (menor estabilidade) |
| `status_propriedade_residencial` | Clientes com hipoteca (Mortgage) têm menor risco; aluguel apresenta risco intermediário |
| `renda_comprovada` | Clientes com renda verificada apresentam menor inadimplência |
| `faturamento_anual` | Correlação inversa esperada — maior renda, menor PD |
| `limite_total_produtos_credito` | Maior limite associado a menor PD |
| `limite_total_rotativos` | Decis mais altos apresentam menor inadimplência |
| `taxa_utilizacao_limite_rotativos` | **Forte discriminação** — alta utilização do limite rotativo eleva significativamente a PD |
| `qt_produtos_credito_contratados_atualmente` | Sem ordenação clara nos decis — maior número de produtos não implica maior risco |
| `qt_produtos_credito_contratados_historicamente` | Decil 0 com maior PD (clientes novos, histórico curto) |
| `qt_registros_publicos_depreciativos` | Presença de registros negativos eleva PD |
| `qt_consultas_credito_6meses` | Clientes sem consultas recentes têm maior risco (menor engajamento) |
| `qt_incidencias_inadimplencia_vencidas_30dias` | Incidências vencidas indicam maior risco |
| `data_contratacao_primeiro_produto_credito` | Clientes mais antigos tendem a oferecer menor risco |
| `qt_meses_desde_ultimo_registro_publico` | Baixo poder discriminativo isolado |
| `qt_meses_classificacao_mais_recente_90dias` | Classificações recentes indicam maior risco |
| `qt_meses_ultima_inadimplencia` | Decil 0 (inadimplências recentes) apresenta maior PD — inadimplências antigas têm efeito diluído |

---

## 11. Feature Engineering

As seguintes transformações foram aplicadas para criar variáveis com maior poder preditivo:

### 11.1 Faixas de Tempo no Emprego Atual

```
qt_anos_mesmo_emprego → {3_YEARS, 6_YEARS, 9_YEARS, 10_YEARS+}
```
Agrupamento em 4 faixas para reduzir dimensionalidade e capturar o comportamento de risco por estabilidade profissional.

### 11.2 Flag de Registros Públicos Negativos

```
registros_publicos_depreciativos → {sem_registros_negativos, com_registros_negativos}
```

### 11.3 Flag de Consultas de Crédito (6 meses)

```
consultas_credito_6meses → {sem_consultas, com_consultas}
```

### 11.4 Comprometimento de Renda Anual

```
comprometimento_de_renda_anual = (pagamento_mensal × qt_parcelas / anos_do_plano) / faturamento_anual × 100
```
Mede a proporção da renda anual comprometida com o empréstimo solicitado — variável criada do zero a partir de dados disponíveis.

### 11.5 Conversão de Datas para Meses Decorridos

```
data_financiamento_emprestimo     → mths_since_data_financiamento_emprestimo
data_contratacao_primeiro_produto → mths_since_data_contratacao_primeiro_produto
```
Referência temporal: 2023-09-20. Nulos e valores negativos imputados pela mediana.

### 11.6 Binarização de Features Categóricas

| Variável | Transformação |
|---|---|
| `qt_parcelas` | 36 meses = 0; 60 meses = 1 |
| `inadimplencia_vencida_30dias` | sem inadimplência = 0; com inadimplência = 1 |
| `tipo_de_concessao_do_credor` | f = 0; w = 1 |
| `plano_de_pagamento` | n = 0; y = 1 |
| `renda_comprovada` | Source Verified = 0; Not Verified = 1 |
| `consultas_credito_6meses` | sem consultas = 0; com consultas = 1 |

### 11.7 Target Encoder — Bad Rate

Variáveis categóricas de alta cardinalidade foram encodadas pela **taxa de inadimplência observada** em cada categoria (calculada exclusivamente no treino). Para cada categoria `c` da variável `X`:

```
X_enc[c] = (qt_bad[c] / qt_total[c]) × 100
```

Encoders salvos em `.csv` para aplicação idêntica na base de teste, prevenindo data leakage.

Variáveis encodadas: `grau_de_emprestimo`, `subclasse_de_emprestimo`, `produto_de_credito`, `qt_anos_mesmo_emprego`, `status_propriedade_residencial`, `estado`, entre outras.

---

## 12. Feature Selection

O processo de seleção de variáveis ocorreu em três estágios sequenciais, aplicados sobre dados imputados (mediana):

**Estágio 1 — Corte de Variância:** Eliminação de variáveis com variância próxima de zero (praticamente constantes), pois não agregam poder discriminativo.

**Estágio 2 — Informação Mútua:** Seleção por Mutual Information entre cada feature e a target binária. A Informação Mútua quantifica a redução de incerteza sobre a target ao se conhecer o valor da feature — variáveis com MI = 0 são descartadas.

**Estágio 3 — Random Forest Feature Importance:** Treinamento de uma Random Forest (`n_estimators=10`, `class_weight` balanceado) para rankear a importância de cada variável pela redução de impureza nas árvores. Apenas features com `feature_importance > 0` avançam para a modelagem.

**Resultado:** 29 features selecionadas para o modelo preditivo.

---

## 13. Modelagem Inicial — Comparativo de Algoritmos

Cinco algoritmos foram treinados e comparados via métricas de treino, teste e validação cruzada (5-fold):

| Modelo | Justificativa |
|---|---|
| **Regressão Logística** | Alta interpretabilidade, estabilidade e baseline sólido |
| **Naive Bayes** | Probabilidades condicionais — baixo custo computacional |
| **KNN Classifier** | Modelo intuitivo baseado em similaridade entre amostras |
| **Random Forest** | Ensemble por Bagging — alta robustez ao overfitting |
| **XGBoost** | Ensemble por Boosting — alto poder preditivo |

### Métricas de Avaliação

As métricas principais adotadas, considerando o impacto financeiro de cada tipo de erro, são:

**Recall (Sensibilidade):** Minimizar Falsos Negativos — clientes inadimplentes aprovados indevidamente representam perda direta para a instituição.

**AUC-ROC:** Avalia a capacidade discriminativa do modelo em todos os thresholds, equilibrando a Taxa de Verdadeiros Positivos e a Taxa de Falsos Positivos.

**KS (Kolmogorov-Smirnov):** Mede a máxima diferença entre as distribuições acumuladas de bons e maus pagadores — quanto maior, melhor a separação entre as classes.

> O **XGBoost** apresentou o melhor desempenho no comparativo inicial e foi selecionado para a etapa de otimização.

---

## 14. Otimização do Modelo Final

### 14.1 Otimização de Hiperparâmetros (BayesSearchCV)

O XGBoost foi otimizado via **Busca Bayesiana** com 5-fold CV e função objetivo de maximização do **Recall**, buscando os melhores valores para:

| Hiperparâmetro | Descrição | Função |
|---|---|---|
| `eval_metric` | LogLoss | Função de custo do treinamento |
| `n_estimators` | Nº de árvores | Capacidade do ensemble |
| `max_depth` | Profundidade máxima | Controle de complexidade |
| `learning_rate` | Taxa de aprendizado | Contribuição de cada árvore |
| `reg_alpha` | Regularização L1 (Lasso) | Esparsidade dos pesos |
| `reg_lambda` | Regularização L2 (Ridge) | Suavização dos pesos |
| `gamma` | Parâmetro de poda | Controle de crescimento das árvores |
| `colsample_bytree` | % de colunas por árvore | Regularização por subamostragem |
| `subsample` | % de linhas por árvore | Regularização por subamostragem |
| `scale_pos_weight` | Peso da classe positiva | Tratamento do desbalanceamento |

> Após a otimização, observou-se melhora expressiva de **Recall, AUC e KS**, com perda marginal de Precision — trade-off desejável dado o contexto de minimização de inadimplência.

### 14.2 Otimização do Threshold de Probabilidade

O threshold padrão (0,50) foi substituído pelo valor que **maximiza o retorno financeiro estimado** sobre a amostra de treino, calculando para cada corte o lucro líquido considerando:

- **Verdadeiro Positivo:** cliente inadimplente corretamente reprovado → perda evitada
- **Falso Positivo:** cliente adimplente incorretamente reprovado → receita perdida
- **Falso Negativo:** cliente inadimplente aprovado → perda efetiva
- **Verdadeiro Negativo:** cliente adimplente aprovado → receita realizada

**Threshold final selecionado: 0,30**

> Clientes com probabilidade de default ≤ 0,30 são aprovados para concessão de crédito. Acima desse limiar, o risco financeiro é considerado inviável.

---

## 15. Política de Crédito (Baseline Rules-Based)

Em complemento ao modelo preditivo, foi desenvolvida uma **Política de Crédito baseada em regras** utilizando os **5 C's do Crédito** como framework conceitual e o **Information Value (IV)** como critério de seleção de variáveis.

### Critério de Corte por IV (segundo Laredo, 2010)

| Faixa de IV | Classificação |
|---|---|
| IV < 0,02 | Ruim — descartada |
| 0,02 ≤ IV < 0,10 | Baixa — descartada |
| 0,10 ≤ IV < 0,30 | Média — **incluída** |
| 0,30 ≤ IV < 0,50 | Alta — **incluída** |
| IV ≥ 0,50 | Extremamente Alta — **incluída** |

Apenas variáveis com **IV ≥ 0,10** foram incluídas na política.

### Variáveis por Dimensão dos 5 C's

**Caráter** — histórico de comportamento e reputação do cliente:
- `qt_meses_classificacao_mais_recente_90dias` (IV = 0,24)
- `qt_meses_ultima_inadimplencia` (IV = 0,18)

**Capacidade** — condições de pagamento e comprometimento de renda:
- `limite_rotativos_utilizado` (IV = 0,50)
- `comprometimento_de_renda_anual` (IV = 0,14)
- `faturamento_anual` (IV = 0,12)

**Colateral** — garantias oferecidas e grau do empréstimo:
- `subclasse_de_emprestimo` (IV = 0,32)
- `grau_de_emprestimo` (IV = 0,29)

**Condições** — contexto econômico e características do produto:
- `taxa_de_juros` (IV = 0,44)

**Capital** — solidez financeira interna da instituição:
- Nenhuma variável identificada na base

### Metodologia de Criação da Política

1. Discretização das variáveis contínuas em **decis**
2. Cálculo da **taxa de inadimplência** (Bad Rate) por categoria/decil
3. Cálculo da **"PD total"** do cliente como soma das PDs individuais de cada variável
4. Definição do threshold de aprovação com base na maximização do retorno financeiro: **threshold = 0,30**

---

## 16. Resultados Consolidados

### Modelo Preditivo — Métricas Comparativas

| Modelo | AUC (Teste) | KS (Teste) | Recall (Teste) | Consistência CV |
|---|---|---|---|---|
| Regressão Logística | Moderada | Moderado | Moderado | ✅ Estável |
| Naive Bayes | Baixa | Baixo | Baixo | ✅ Estável |
| KNN Classifier | Moderada | Moderado | Moderado | ⚠️ Variação |
| Random Forest | Alta | Alto | Alto | ✅ Estável |
| **XGBoost** | **Mais Alta** | **Mais Alto** | **Mais Alto** | **✅ Estável** |
| **XGBoost (Bayes Search)** | **Melhor** | **Melhor** | **Melhor** | **✅ Estável** |

> As métricas consistentes entre treino, teste e validação cruzada confirmam boa generalização e ausência de overfitting.

## 16. Resultados Consolidados

### 16.1 Comparativo de Performance dos Modelos

Os modelos foram avaliados utilizando métricas clássicas de risco de crédito, com foco principal em:

- Capacidade discriminatória (`AUC`)
- Separação entre GOOD e BAD (`KS`)
- Capacidade de captura de inadimplentes (`Recall`)
- Estabilidade temporal
- Performance financeira

| Modelo | AUC | KS | Recall | Estabilidade | Observações |
|---|---|---|---|---|---|
| Regressão Logística | Moderado | Moderado | Moderado | Alta | Baseline interpretável |
| Naive Bayes | Baixo | Baixo | Baixo | Alta | Pouca capacidade discriminatória |
| KNN Classifier | Moderado | Moderado | Moderado | Média | Sensível à dimensionalidade |
| Random Forest | Alto | Alto | Alto | Alta | Bom desempenho geral |
| XGBoost | Muito Alto | Muito Alto | Muito Alto | Alta | Melhor modelo base |
| **XGBoost + BayesSearchCV** | **Melhor Resultado** | **Melhor Resultado** | **Melhor Resultado** | **Alta** | **Modelo Final Selecionado** |

> O XGBoost otimizado apresentou a melhor combinação entre performance estatística, estabilidade temporal e capacidade de ordenação de risco.

---

### 16.2 Indicadores Consolidados do Modelo Final

| Indicador | Resultado |
|---|---|
| Modelo Selecionado | XGBoost + BayesSearchCV |
| Tipo de Problema | Credit Scoring |
| Variável Target | GOOD / BAD |
| Threshold de Aprovação | 0,30 |
| Métrica Primária | AUC + KS + Recall |
| Estratégia de Balanceamento | `scale_pos_weight` + `class_weight` |
| Estratégia de Validação | Holdout + Cross Validation |
| Distribuição da Target | ~88,7% GOOD / ~11,3% BAD |
| Técnica de Explicabilidade | SHAP Values |

---

### 16.3 Comparativo — Política Atual vs Modelo Preditivo

A política atual baseada em regras foi comparada diretamente com o modelo preditivo de Machine Learning.

A análise demonstrou que o modelo apresentou:

- Melhor separação entre clientes GOOD e BAD
- Menor concentração de inadimplentes nas faixas aprovadas
- Melhor ordenação de risco
- Redução da perda esperada da carteira
- Aprovação mais eficiente ajustada ao risco

| Métrica | Política Atual | Modelo ML |
|---|---|---|
| AUC | Menor | Maior |
| KS | Menor | Maior |
| BAD Médio Carteira Aprovada | Maior | Menor |
| Separação entre Ratings | Moderada | Forte |
| Capacidade de Ordenação de Risco | Limitada | Elevada |
| Perda Esperada (EL) | Maior | Menor |
| Eficiência de Aprovação | Menor | Maior |

> O modelo apresentou capacidade superior de discriminação e controle de risco quando comparado à política tradicional rules-based.

---

### 16.4 Comparativo de Rating — Política vs Modelo

A comparação entre os ratings da política atual e os ratings gerados pelo modelo demonstrou que o modelo apresentou **maior capacidade de ordenação de risco**, concentrando proporcionalmente mais clientes inadimplentes (`BAD`) nas piores faixas de rating e reduzindo a inadimplência nas melhores faixas.

| Rating | % BAD Política | % BAD Modelo |
|---|---|---|
| 0 | 22,69% | **29,10%** |
| 1 | 17,53% | **19,64%** |
| 2 | **14,92%** | 14,79% |
| 3 | **13,73%** | 12,65% |
| 4 | **11,39%** | 10,64% |
| 5 | **10,05%** | 8,34% |
| 6 | **8,28%** | 6,80% |
| 7 | **6,44%** | 5,49% |
| 8 | **5,04%** | 3,54% |
| 9 | **2,84%** | 1,87% |

Principais Evidências:

- O modelo aumentou significativamente a concentração de clientes BAD nos ratings de maior risco (`0` e `1`)
- As melhores faixas de rating apresentaram redução relevante do `%BAD`
- O modelo apresentou comportamento monotônico mais consistente
- Houve melhora na separação entre clientes de baixo e alto risco
- O modelo reduziu a inadimplência nas faixas aprovadas da carteira

O comportamento observado indica que o modelo possui maior capacidade de:

- identificar clientes efetivamente arriscados
- proteger a carteira aprovada
- melhorar a segmentação de risco
- reduzir perdas financeiras associadas à inadimplência

> Em modelos de crédito, concentrar maiores taxas de BAD nas piores faixas de rating é uma evidência de maior capacidade discriminatória e melhor ordenação de risco. Sendo assim, a monotonicidade do BAD por rating é uma das principais evidências de robustez em modelos de risco de crédito.

| Indicador                            | Política Atual | Modelo ML |
| ------------------------------------ | -------------- | --------- |
| Capacidade de Identificação de BAD   | Moderada       | Superior  |
| Concentração de BAD em Ratings Altos | Maior          | Menor     |
| BAD nas Faixas Aprovadas             | Maior          | Menor     |
| Qualidade Média da Carteira          | Moderada       | Superior  |
| Exposição ao Risco                   | Maior          | Menor     |
| Perda Esperada da Carteira           | Maior          | Menor     |


---

## 17. Artefatos Gerados

| Artefato | Localização | Descrição |
|---|---|---|
| `credit_risk_loan_data.parquet` | `data/` | Base de dados principal |
| `features_selected.csv` | `features/` | Features selecionadas com suas importâncias |
| `{feature}_enc.csv` | `features/` | Target encoders (Bad Rate) por variável categórica |
| `modelo_oficial.pkl` | `models/` (implícito) | Modelo XGBoost otimizado treinado |
| `{feature}_enc.csv` | `Credit_Policy/` | Encoders específicos da Política de Crédito |
| `functions.py` | raiz | Módulo central com todas as funções do projeto |

---

## 18. Referências Bibliográficas

[1] LAREDO SICSÚ, Abraham. **Credit Scoring: Desenvolvimento, Implantação e Acompanhamento**. São Paulo: Blucher, 2010. Disponível em: https://www.blucher.com.br/credit-scoring_9788521205333. Acesso em: 22 jan. 2023.

[2] SEBBEN, Renivaldo José. **Análise de Crédito e Cobrança: Como Conceder Crédito com Segurança e Recuperar Créditos Inadimplentes**. Novatec Editora Ltda, 2020. Disponível em: https://novatec.com.br/livros/analise-credito-cobranca. Acesso em: 12 mar. 2023.

[3] GUIMARÃES XAVIER, Caroline. **Risco na Análise de Crédito**. Trabalho de Conclusão de Curso (Bacharelado em Ciências Contábeis) — Departamento de Ciências Contábeis, Universidade Federal de Santa Catarina. Florianópolis, p. 70, 2011. Acesso em: 15 fev. 2023.

[4] JORGE CHAIA, Alexandre. **Modelos de Gestão do Risco de Crédito e sua Aplicabilidade ao Mercado Brasileiro**. Dissertação (Mestrado em Administração) — Faculdade de Economia, Administração e Contabilidade, Universidade de São Paulo. São Paulo, p. 126, 2003. Acesso em: 15 fev. 2023.

[5] ARAÚJO, Elaine Aparecida; MONTREUIL CARMONA, Charles Ulises de. **Desenvolvimento de Modelos Credit Scoring com Abordagem de Regressão Logística para a Gestão da Inadimplência de uma Instituição de Microcrédito**. *Contabilidade Vista & Revista*, Minas Gerais, v. 18, n. 3, p. 107–131, set. 2007. Acesso em: 20 fev. 2023.

[6] SHELCI SILVA, Juelline. **Gerenciamento Integrado de Riscos: Modelos de Predição de Risco de Crédito em Machine Learning para a Identificação de Ativos Problemáticos em uma Instituição Financeira**. Dissertação (Mestrado Profissional em Economia) — Faculdade de Administração Contabilidade e Economia, Universidade de Brasília. Brasília, p. 74, 2022. Acesso em: 20 fev. 2023.

[7] FORTI, Melissa. **Técnicas de Machine Learning Aplicadas na Recuperação de Crédito do Mercado Brasileiro**. Dissertação (Mestrado em Economia) — Escola de Economia de São Paulo, Fundação Getulio Vargas. São Paulo, 2018. Acesso em: 21 fev. 2023.

[8] SANTOS, Patrick Ferreira dos. **Uso de Técnicas de Machine Learning para Análise de Risco de Crédito**. Dissertação (Mestrado Profissional em Economia) — Faculdade de Administração Contabilidade e Economia, Universidade de Brasília. Brasília, p. 57, 2022. Acesso em: 19 fev. 2023.

[9] ARAÚJO, João Paulo Bezerra de. **Interpretabilidade de Modelos de Machine Learning: Aplicação no Mercado de Crédito**. Trabalho de Conclusão de Curso (Bacharelado em Engenharia Elétrica) — Universidade Federal do Ceará. Fortaleza, p. 73, 2020. Acesso em: 19 fev. 2023.

[10] MONTOYA, Anna; ODINTSOV, Kirill; KOTEK, Martin. **Home Credit Default Risk**. Kaggle Competition. Disponível em: https://www.kaggle.com/competitions/home-credit-default-risk/overview. Acesso em: 10 jan. 2023.

[11] IZBICKI, Rafael; DOS SANTOS, Tiago Mendonça. **Aprendizado de Máquina: Uma Abordagem Estatística**. São Carlos: Câmara Brasileira do Livro, 2020. Disponível em: https://loja.uiclap.com/titulo/ua24032. Acesso em: 23 fev. 2023.

[12] GÉRON, Aurélien. **Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow: Concepts, Tools and Techniques to Build Intelligent Systems**. O'Reilly, 2019. Disponível em: https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632. Acesso em: 25 jan. 2023.

[13] HARRISON, Matt. **Machine Learning Pocket Reference**. O'Reilly, 2019. Disponível em: https://www.oreilly.com/library/view/machine-learning-pocket/9781492047537. Acesso em: 27 jan. 2023.

[14] MORETTIN, Pedro A.; BUSSAB, Wilton De O. **Estatística Básica**. São Paulo: Saraiva, 2017. Disponível em: https://www.saraiva.com.br/estatistica-basica---morettin---saraiva-21397/p. Acesso em: 10 jan. 2023.

---