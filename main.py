from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate

import config
import streamlit as st
import random

CHROMA_PATH = "chroma"
DATA_PATH = "Documents"
embedding_function = OllamaEmbeddings(model=config.embeddings_model, base_url=config.base_url)
llm = ChatOllama(model=config.chat_model, base_url=config.base_url, keep_alive='10m')
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

st.title("Chat with docs")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [("system", '{sys_prompt}')]

for message in st.session_state['messages']:
    message_role = message[0] if message[0] == 'user' or 'system' else 'assistant'
    if message_role == 'system':
        continue
    st.chat_message(message[0]).markdown(message[1])

if prompt := st.chat_input('Message llm..'):
    st.chat_message('user').markdown(prompt)
    st.session_state["messages"].append(('user', prompt))

    results = db.similarity_search_with_relevance_scores(prompt, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        context_text = ""
    else:
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        print(context_text)
    prompt_template = ChatPromptTemplate.from_messages(st.session_state['messages'])
    chain = prompt_template | llm 

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream({"sys_prompt": "Answer the question based only on the following context: \n\n" if context_text else "Be a helpful AI assistant"}):
            full_response += chunk.content
            message_placeholder.markdown(full_response.rstrip('<|im_end|>') + random.choice(["⬤", "●"]))

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        extra_text = f"\n\nSources: ```{sources}```" if sources else ''
        formatted_response = f"{full_response.rstrip('<|im_end|>')}" + extra_text
        message_placeholder.markdown(formatted_response)

    st.session_state.messages.append(('ai', full_response))