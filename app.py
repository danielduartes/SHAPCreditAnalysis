import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load('models/credit_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.title('💳 Análise de Risco de Crédito')
st.write('Preencha os dados do cliente para prever o risco de inadimplência.')

col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Idade', min_value=18, max_value=100)
    credit_amount = st.number_input('Valor do crédito', min_value=0, max_value=20000)
    duration = st.number_input('Duração (meses)', min_value=1, max_value=72)
    job = st.selectbox('Qualificação profissional', ['highly skilled', 'skilled', 'unskilled non-resident', 'unskilled resident'])
    housing = st.selectbox('Moradia', ['own', 'free', 'rent'])

with col2:
    saving_accounts = st.selectbox('Poupança', ['little', 'moderate', 'quite rich', 'rich', 'none'])
    checking_account = st.selectbox('Conta corrente', ['little', 'moderate', 'rich', 'none'])
    purpose = st.selectbox('Finalidade do crédito', ['business', 'car', 'domestic appliances', 'education', 'furniture/equipment', 'radio/TV', 'repairs', 'vacation/others'])

if st.button('Analisar Risco'):
    # Monta o dicionário zerado
    input_dict = {col: 0 for col in [
        'Age', 'Credit amount', 'Duration',
        'Job_highly skilled', 'Job_skilled', 'Job_unskilled non-resident', 'Job_unskilled resident',
        'Housing_free', 'Housing_own', 'Housing_rent',
        'Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_none',
        'Saving accounts_quite rich', 'Saving accounts_rich',
        'Checking account_little', 'Checking account_moderate', 'Checking account_none', 'Checking account_rich',
        'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education',
        'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others'
    ]}

    # Preenche os valores
    input_dict['Age'] = age
    input_dict['Credit amount'] = np.log1p(credit_amount)
    input_dict['Duration'] = duration
    input_dict[f'Job_{job}'] = 1
    input_dict[f'Housing_{housing}'] = 1
    input_dict[f'Saving accounts_{saving_accounts}'] = 1
    input_dict[f'Checking account_{checking_account}'] = 1
    input_dict[f'Purpose_{purpose}'] = 1

    # Escala e prediz
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)

    proba = model.predict_proba(input_scaled)[0]
    pred = model.predict(input_scaled)[0]

    st.divider()

    if pred == 0:
        st.error(f'⚠️ Alto risco de inadimplência! Probabilidade de mau pagador: {proba[0]:.1%}')
    else:
        st.success(f'✅ Baixo risco! Probabilidade de bom pagador: {proba[1]:.1%}')

    # SHAP
    st.subheader('📊 Explicação da decisão')
    explainer = shap.LinearExplainer(model, input_scaled_df)
    shap_values = explainer(input_scaled_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
