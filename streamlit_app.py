import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import plotly.express as px
import streamlit as st
from src.build_vector_db import build_db
from src.rag_engine import answer_question

st.set_page_config(page_title='FIFA World Cup RAG Assistant', layout='wide')
st.title('🏆 FIFA World Cup RAG Assistant: 1930–2022')
st.caption('Ask questions, compare tournaments, and explore winners, best players, goalkeepers and top scorers.')

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'worldcup_summary.csv')
df = pd.read_csv(DATA_PATH)

if st.sidebar.button('Build / Refresh Vector DB'):
    build_db()
    st.sidebar.success('Vector database updated.')

with st.sidebar:
    st.header('Filters')
    years = st.multiselect('Select years to compare', df['year'].tolist(), default=[2014, 2018, 2022])

st.subheader('Dataset')
st.dataframe(df, use_container_width=True)

st.subheader('Compare Football World Cups')
compare_df = df[df['year'].isin(years)] if years else df.tail(5)
st.dataframe(compare_df[['year','host','winner','runner_up','best_player','best_goalkeeper','top_scorer','top_scorer_goals']], use_container_width=True)

fig = px.bar(compare_df, x='year', y='top_scorer_goals', color='winner', title='Top Scorer Goals by Selected Tournament')
st.plotly_chart(fig, use_container_width=True)

st.subheader('Ask the RAG Assistant AI')
query = st.text_input('Example: Compare 1986 and 2022 World Cups')
if st.button('Write a query above ') and query:
    with st.spinner('Retrieving from vector database and generating answer...'):
        st.write(answer_question(query, k=6))

st.subheader('Approximate Budget')
budget = pd.DataFrame([
    ['Data collection and cleaning', '1–2 days', '£150–£400'],
    ['RAG pipeline + vector database', '2–3 days', '£400–£800'],
    ['Streamlit dashboard', '2 days', '£300–£600'],
    ['LLM integration and testing', '1–2 days', '£200–£500'],
    ['Deployment and documentation', '1 day', '£150–£300'],
])
budget.columns = ['Task', 'Time', 'Approximate Cost']
st.table(budget)
st.info('Total student/portfolio budget: £0–£100 using free/local tools. Freelance production estimate: £1,200–£2,600.')
