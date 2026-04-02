# 💳 German Credit Risk Analysis

Modelo de machine learning para classificar clientes como bons ou maus pagadores, com foco em **explicabilidade das decisões** usando SHAP — tornando o modelo auditável e interpretável para o negócio.

---

## 🎯 Objetivo

Instituições financeiras precisam não só prever inadimplência, mas **justificar** cada decisão de crédito — seja por exigência regulatória ou por transparência com o cliente. Este projeto combina um modelo preditivo com explicabilidade individual via SHAP, respondendo: *"Por que esse cliente foi classificado como risco?"*

---

## 📦 Dataset

- **Fonte:** [German Credit Data — Kaggle](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)
- **Tamanho:** 1.000 clientes, 10 variáveis
- **Target:** `Risk` — `good` (bom pagador) ou `bad` (mau pagador) — classe desbalanceada: ~70% good

---

## 🗂️ Estrutura do Projeto

```
credit-risk-analysis/
│
├── data/
│   ├── german_credit_data.csv    # dataset original
│   └── credit_limpo.csv          # dataset após limpeza
│
├── models/
│   ├── credit_model.pkl          # modelo treinado
│   └── scaler.pkl                # scaler
│
├── credit_eda.ipynb              # análise exploratória
├── credit_ml.ipynb               # modelagem, avaliação e SHAP
├── app.py                        # interface Streamlit
└── requirements.txt
```

---

## 🔍 Análise Exploratória (EDA)

- Boxplots de `Age`, `Duration` e `Credit amount`
- KDE de `Credit amount` por `Risk` — distribuição de valores por perfil de risco
- Scatter plot de `Credit amount` vs `Duration` colorido por `Risk`
- Matriz de correlação completa após encoding das variáveis categóricas

**Principais insights:**
- Clientes **sem conta corrente** têm o maior risco de inadimplência
- A combinação de **crédito alto + prazo longo** concentra a maioria dos maus pagadores
- `Credit amount` tem distribuição assimétrica positiva forte — transformação logarítmica aplicada
- `Saving accounts` e `Checking account` são as variáveis mais correlacionadas com `Risk`

---

## 🤖 Modelagem

### Pré-processamento
- Tratamento de valores nulos em `Saving accounts` e `Checking account` com categoria `none`
- Transformação logarítmica em `Credit amount` para reduzir impacto de outliers
- Encoding de variáveis categóricas com `pd.get_dummies`
- Remoção da coluna `Sex` por baixa correlação e questões éticas de viés
- Normalização com `StandardScaler`
- Divisão treino/teste: 80/20 com `stratify=y`

### Modelos Treinados e Comparação

| Modelo | Acurácia | Recall Mau Pagador | Maus Pagadores Perdidos |
|---|---|---|---|
| Regressão Logística | 71% | 40% | 36 |
| **Regressão Logística (balanced)** | **68%** | **78%** | **13** |
| XGBoost | 74% | 60% | 24 |
| Random Forest (balanced) | 75% | 45% | 33 |

### Modelo Final
**Regressão Logística com `class_weight='balanced'`**

Em análise de crédito, o custo de aprovar um mau pagador é muito maior do que recusar um bom pagador. Por isso, o **Recall da classe negativa** é a métrica prioritária — e a Regressão Logística balanceada deixa escapar apenas 13 maus pagadores, contra 36 do modelo sem balanceamento.

```
              precision    recall  f1-score
           0       0.48      0.78      0.59
           1       0.87      0.64      0.74
    accuracy                           0.68
```

---

## 🔎 Explicabilidade com SHAP

O diferencial deste projeto é a explicabilidade individual de cada previsão usando **SHAP (SHapley Additive exPlanations)**.

### Importância Global (Bar plot)
As variáveis mais impactantes no modelo:
1. `Checking account_none` — não ter conta corrente é o maior fator de risco
2. `Duration` — prazos longos aumentam o risco
3. `Checking account_little` — pouco saldo na conta corrente
4. `Age` — idade tem efeito complexo e não linear
5. `Checking account_moderate`

### Direção do Impacto (Beeswarm)
- Clientes **sem conta corrente** têm risco significativamente aumentado
- **Prazos longos** aumentam o risco de inadimplência
- **Conta corrente rica** reduz fortemente o risco

### Explicação Individual (Waterfall)
Para cada cliente, é possível visualizar exatamente **quais variáveis empurraram a decisão** para bom ou mau pagador — tornando o modelo auditável e justificável.

---

## 🚀 Interface — Streamlit

A aplicação permite inserir os dados de um cliente e retorna o risco de inadimplência junto com o **gráfico SHAP waterfall** explicando a decisão.

**Como rodar:**

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🛠️ Tecnologias

- Python 3.11
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn
- SHAP
- Streamlit
- Joblib
