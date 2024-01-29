from langchain.chat_models import ChatOpenAI
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from io import BytesIO
from tempfile import NamedTemporaryFile
import os
from langchain.vectorstores import FAISS

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
def chat_complete(user_prompt, model="gpt-3.5-turbo", temperature=0.7):
    """
    This function generates text based on user input.

    Args:
        user_prompt (string): User input
        temperature (float): Value between 0 and 1. Defaults to 0.7
        model (string): "gpt-3.5-turbo" or "gpt-4".

    Return:
        generated text

    All the conversations are stored in st.session_state variables.
    """

    openai_llm = ChatOpenAI(
        openai_api_key=st.session_state.openai_api_key,
        temperature=temperature,
        model_name=model,
        streaming=True,
        callbacks=[StreamHandler(st.empty())]
    )

    # Add the user input to the messages
    st.session_state.messages.append(HumanMessage(content=user_prompt))
    try:
        response = openai_llm(st.session_state.messages)
        generated_text = response.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    if generated_text is not None:
        # Add the generated output to the messages
        st.session_state.messages.append(response)

    return generated_text



def summarize_document(docs, model="gpt-3.5-turbo"):

    map_prompt_template = """
        Write a summary of this chunk of text that includes the main points and any important details.
        {text}
        """

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    combine_prompt_template = """
        Write a concise summary of the following text delimited by triple backquotes.
        Return your response in bullet points which covers the key points of the text.
        ```{text}```
        SUMMARY:
        """

    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text"]
    )

    openai_llm = ChatOpenAI(
        openai_api_key=st.session_state.openai_api_key,
        temperature=0,
        model_name=model,
        streaming=True
    )
    summary_chain = load_summarize_chain(
        llm=openai_llm,
        # retriever=vector_store.as_retriever(),
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt,
        return_intermediate_steps=False,
        verbose=True,
    )
    output = summary_chain.run(docs)
    print(output)
    
    return output


def get_vector_store(uploaded_file):
    """
    This function takes an UploadedFile object as input,
    and returns a FAISS vector store.
    """
    
    with st.sidebar:
        st.write("**Summarize Text**")
        summary_option = st.radio(
            label="Summarize Text",
            options=("Enabled", "Disabled"),
            label_visibility="collapsed",
            index=1
        )

    if uploaded_file is None:
        return None

    file_bytes = BytesIO(uploaded_file.read())

    # Create a temporary file within the "files/" directory
    with NamedTemporaryFile(dir="files/", delete=False) as file:
        filepath = file.name
        file.write(file_bytes.read())

    # Determine the loader based on the file extension.
    if uploaded_file.name.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif uploaded_file.name.lower().endswith(".txt"):
        loader = TextLoader(filepath)
    elif uploaded_file.name.lower().endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif uploaded_file.name.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
    elif uploaded_file.name.lower().endswith(".html"):
        loader = UnstructuredHTMLLoader(filepath)
    elif uploaded_file.name.lower().endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(filepath)
    else:
        st.error("Please load a file in pdf or txt", icon="🚨")
        if os.path.exists(filepath):
            os.remove(filepath)
        return None

    # Load the document using the selected loader.
    document = loader.load()

    try:
        with st.spinner("Vector store in preparation..."):
            # Split the loaded text into smaller chunks for processing.
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                # separators=["\n", "\n\n", "(?<=\. )", "", " "],
            )
            doc = text_splitter.split_documents(document)
            # Create a FAISS vector database.
            embeddings = OpenAIEmbeddings(
                openai_api_key=st.session_state.openai_api_key
            )
            vector_store = FAISS.from_documents(doc, embeddings)
            if summary_option == "Enabled":
                st.write(summarize_document(doc))
    except Exception as e:
        vector_store = None
        st.error(f"An error occurred: {e}", icon="🚨")
    finally:
        # Ensure the temporary file is deleted after processing
        if os.path.exists(filepath):
            os.remove(filepath)

    return vector_store


def document_qna(query, vector_store, model="gpt-3.5-turbo"):
    """
    This function takes a user prompt, a vector store and a GPT model,
    and returns a response on the uploaded document along with sources.
    """

    if vector_store is not None:
        if st.session_state.memory is None:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )

        openai_llm = ChatOpenAI(
            openai_api_key=st.session_state.openai_api_key,
            temperature=0,
            model_name=model,
            streaming=True,
            callbacks=[StreamlitCallbackHandler(st.empty())]
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=openai_llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            # retriever=vector_store.as_retriever(search_type="mmr"),
            memory=st.session_state.memory,
            return_source_documents=True
        )

        try:
            # response to the query is given in the form
            # {"question": ..., "chat_history": [...], "answer": ...}.
            response = conversation_chain({"question": query})
            generated_text = response["answer"]
            source_documents = response["source_documents"]

        except Exception as e:
            generated_text, source_documents = None, None
            st.error(f"An error occurred: {e}", icon="🚨")
    else:
        generated_text, source_documents = None, None

    return generated_text, source_documents
