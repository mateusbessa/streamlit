import streamlit as st
import pandas as pd
import joblib as jb
import numpy as np

model = jb.load('model_portabilidade.pkl')

st.title('Análise de provável evasão de clientes')

with st.form(key="data_client"):
    name = st.text_input(label='Nome do Cliente')
    credit_score = st.slider('Score',0,2000)
    tenure = st.number_input(label='Tempo de relacionamento com o banco',min_value=0,max_value=100)
    balance = st.number_input(label='Saldo em conta')
    num_of_products = st.number_input('Produtos adquiridos',min_value=0,max_value=99)
    has_crcard = st.checkbox(label='Tem cartão de crédito')
    is_active_member = st.checkbox(label='Membro ativo')
    estimated_salary = st.slider('Salário',0,200000)
    input_buttom = st.form_submit_button("Analisar")
    test = np.array([[credit_score,tenure, balance,num_of_products,has_crcard,is_active_member,estimated_salary]])
    classify = model.predict(test)
    result = model.predict_proba(test)
 

if input_buttom:
    if classify == 1:
        st.write('### É provável que o cliente nos abandone')
    elif classify == 0:
        st.write('### Não é provável que o cliente saia')
    st.write(f"### Probabilidade do cliente abandonar é de {str([result[:,1]*100])[8:12]}%")