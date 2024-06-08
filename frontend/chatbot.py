import streamlit as st
import pandas as pd
import sys
import re

# Insert path for custom modules
sys.path.insert(1, '/home/jabez/Documents/week_7/Precision-RAG/scripts')
import query_data
import answer

# Initialize message lists
user_messages = []
chatbot_messages = []

# Display title and initial greeting
st.write("""
# Ethiopian Criminal Law Expert Chatbot
""")

# Display initial chatbot message
with st.chat_message(name='assistant'):
    initial_message = "Please ask me a question about Ethiopian Criminal Law"
    chatbot_messages.append(initial_message)
    st.write(initial_message)

# Chat input logic
if prompt := st.chat_input("Say something"):
    # Append user input to list
    user_messages.append(prompt)

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_messages[-1])

    try:
        # Call the chatbot logic function and get the response
        response = query_data.process_message(prompt)
    except Exception as e:
        response = f"Sorry, there was an error processing your message: {e}"
    
    chatbot_messages.append(response)

    # Display chatbot response in chat message container
    with st.chat_message(name='assistant'):
        st.write(chatbot_messages[-1])

    # Extract prompts from the response
    prompts = re.findall(r'Prompt \d+: (.*?)(?=\nPrompt \d+:|\nSources:)', response, re.DOTALL)
    data = []
    for prompt in prompts:
        answers =  answer.answer_message(prompt)
        data.append({'question': prompt.strip(), 'answer': answers})
    df = pd.DataFrame(data)
    # st.write(df)
    from datasets import Dataset
    dataset = Dataset.from_dict(df)
    from ragas import evaluate

    from ragas.metrics import (
        answer_relevancy,
        context_recall,
    )
    result = evaluate(
        dataset = dataset,
        metrics=[
            
            answer_relevancy,
            
        ],
    )
    eva_an = result.to_pandas()
    sorted_df = eva_an.sort_values(by='answer_relevancy', ascending=True)
    st.write(sorted_df)