import streamlit as st
import pandas as pd
import sys
sys.path.insert(1, '/home/jabez/Documents/week_7/Precision-RAG/scripts')
import query_data
# Create empty lists to store messages
user_messages = []
chatbot_messages = []

# Display title and initial greeting
st.write("""
# Ethiopian Criminal Law Expert Chatbot
""")
# def chatbot_logic(user_input):
#         response = query_data.process_message(user_input)
#         return response

# Display initial chatbot message
with st.chat_message (name='assistant'):
    chatbot_messages.append("Please ask me a question about the Ethiopian Criminal law")
    st.write(chatbot_messages[-1])  # Access the latest message

if prompt := st.chat_input("Say something"):
    # Append user input to list
    user_messages.append(prompt)

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_messages[-1])  # Access the latest message

    # Call the chatbot logic function and get the response
    response = query_data.process_message(prompt)
    chatbot_messages.append(response)
    with st.chat_message (name='assistant'):
        st.write(chatbot_messages[-1])  # Access the latest message


