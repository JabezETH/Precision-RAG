# Precision-RAG

Prompt Tuning For Building Enterprise Grade RAG Systems

## Business objective

PromptlyTech is an innovative e-business specializing in providing AI-driven solutions for optimizing the use of Language Models (LLMs) in various industries. The company aims to revolutionize how businesses interact with LLMs, making the technology more accessible, efficient, and effective. By addressing the challenges of prompt engineering, the company plays a pivotal role in enhancing decision-making, operational efficiency, and customer experience across various industries. PromptlyTech's solutions are designed to cater to the evolving needs of a digitally-driven business landscape, where speed and accuracy are key to staying competitive.

## Usage

Run the frontend:
```sh
cd Precision-RAG
cd frontend
streamlit run chatbot.py
```

## How it works

The user provides a question and a scenario related to Ethiopian criminal law. The language model (LLM) then generates a prompt based on the user's input and formulates a response using the given context of Ethiopian criminal law.The original question provided by the user, the scenario, and the LLM's generated response are then used to create an evaluation dataset.Finally, the evaluation dataset is assessed using the RAGA (Relevance, Accuracy, Grammaticality, Appropriateness) toolkit. RAGA analyzes the dataset and provides scores for the relevance, accuracy, grammaticality, and appropriateness of the LLM's response. This workflow allows for the creation of an evaluation dataset that can be used to assess the performance of the language model in answering questions within the specific context of Ethiopian criminal law.

## Installation
setup virtual enviroment: 
```sh
cd Precision-RAG
python3 venv env
source env/bin/actiavte
```
Install the dependencies: 
```sh
pip install -r requirements.txt
```
