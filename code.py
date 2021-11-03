import streamlit as st
import pandas as pd
import joblib as jb
import numpy as np
from streamlit.type_util import Key
from PIL import Image
#import SessionState as ss

model = jb.load('model_portabilidade.pkl')
image = Image.open('alert.png')

#st.beta_set_page_config(page_title='Predição de Churn')

st.title('Análise de provável evasão de clientes')

with st.form(key="data_client"):
    name = st.text_input(label='Nome do Cliente')
    credit_score = st.number_input(label='Score',min_value=0,max_value=2000)
    tenure = st.number_input(label='Tempo de relacionamento com o banco (anos)',min_value=0,max_value=100)
    balance = st.number_input(label='Saldo em conta - U$')
    num_of_products = st.number_input(label='Produtos adquiridos. Ex: Previdência, Seguro, etc.',min_value=0,max_value=99)
    has_crcard = st.checkbox(label='Tem cartão de crédito')
    is_active_member = st.checkbox(label='Membro ativo')
    estimated_salary = st.number_input(label='Salário anual - U$',min_value=0,max_value=200000)
    input_buttom = st.form_submit_button('Analisar')
    test = np.array([[credit_score,tenure, balance,num_of_products,has_crcard,is_active_member,estimated_salary]])
    classify = model.predict(test)
    result = model.predict_proba(test)

if input_buttom:
    if classify == 1:
        st.write('### É provável que o cliente nos abandone')
        st.image(image, output_format='png')
    elif classify == 0:
        st.write('### Não é provável que o cliente abandone')
    st.write(f"### Probabilidade do cliente abandonar é de {str([result[:,1]*100])[8:12]}%")