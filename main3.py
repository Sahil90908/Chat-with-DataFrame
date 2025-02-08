import os
import pandas as pd
import streamlit as st
import google.generativeai as gen_ai
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit web app config
st.set_page_config(
    page_title="DF Chat",
    page_icon="💬",
    layout="centered"
)

GOOGLE_API_KEY = os.getenv("AIzaSyBNcDbmJz0CSY_eWj3-kEGFOCi8cY6NgZQ")

# Configure Gemini API
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel("gemini-pro")

def read_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Streamlit page title
st.title("🤖 DataFrame ChatBot - Gemini")

# Initialize chat history in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initiate df in session state
if "df" not in st.session_state:
    st.session_state.df = None

# File upload widget
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    st.session_state.df = read_data(uploaded_file)
    st.write("DataFrame Preview:")
    st.dataframe(st.session_state.df.head())

    # Select visualization type
    viz_type = st.selectbox("Select Visualization Type", ["None", "Bar Chart", "Histogram", "Scatter Plot"])
    
    if viz_type != "None":
        columns = st.session_state.df.select_dtypes(include=['number']).columns.tolist()
        if viz_type in ["Bar Chart", "Histogram"]:
            selected_column = st.selectbox("Select Column", columns)
        else:
            x_column = st.selectbox("Select X-axis Column", columns)
            y_column = st.selectbox("Select Y-axis Column", columns)
        
        fig, ax = plt.subplots()
        if viz_type == "Bar Chart":
            sns.barplot(x=st.session_state.df[selected_column].value_counts().index,
                        y=st.session_state.df[selected_column].value_counts().values, ax=ax)
            ax.set_title(f"Bar Chart of {selected_column}")
        elif viz_type == "Histogram":
            sns.histplot(st.session_state.df[selected_column], kde=True, ax=ax)
            ax.set_title(f"Histogram of {selected_column}")
        elif viz_type == "Scatter Plot":
            sns.scatterplot(x=st.session_state.df[x_column], y=st.session_state.df[y_column], ax=ax)
            ax.set_title(f"Scatter Plot between {x_column} and {y_column}")
        
        st.pyplot(fig)

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for user's message
user_prompt = st.chat_input("Ask LLM...")

if user_prompt and st.session_state.df is not None:
    # Add user's message to chat history and display it
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    # Process question related to DataFrame
    df_summary = st.session_state.df.describe().to_string()
    prompt = f"The user uploaded a dataset and asked: {user_prompt}. Here is a summary of the dataset:\n{df_summary}\nAnswer the question based on the data."

    # Get response from Gemini API
    gemini_response = model.generate_content(prompt)
    assistant_response = gemini_response.text

    st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

    # Display LLM response
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
