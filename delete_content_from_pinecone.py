import streamlit as st
import os
import sys
import time
import openai
import pinecone
import traceback
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings

# https://docs.pinecone.io/docs/metadata-filtering

try:
    import environment_variables
except ImportError:
    pass

try:
    # Setting page title and header
    st.set_page_config(
        page_title="Delete Pinecone.io data", page_icon=":robot_face:")

    # Step 1: Get common environment variables
    OPENAI_API_KEY = os.environ['openai_api_key']
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # container for text box
    container = st.container()

    with container:
        st.title("Beware! You are deleting content from Pinecone.io")

        with st.form("problem-statement-form"):
            database_name = st.selectbox('Select a database:', (
                '', 'problem-statements', 'langchain-ttsh', 'market-solutions'), key="contributor_list")
            source = st.text_input('Source:', key='source',)

            is_submitted = st.form_submit_button(
                label="Delete", )

            if is_submitted:
                database_name = database_name.strip()
                source = source.strip()

                if database_name == "":
                    st.error("Please select a database.")
                elif source == "":
                    st.error("Please write a source name.")
                else:
                    if database_name == "problem-statements":
                        # Databaes 1: Problem Statement Pinecone.io Database
                        # =======================================================
                        problem_statement_pinecone_api_key = os.environ[
                            'problem_statement_pinecone_api_key']
                        problem_statement_pinecone_environment = os.environ[
                            'problem_statement_pinecone_environment']
                        problem_statement_index_name = os.environ['problem_statement_index_name']

                        # Initialize connection to pinecone (get API key at app.pinecone.io)
                        pinecone.init(
                            api_key=problem_statement_pinecone_api_key,
                            environment=problem_statement_pinecone_environment
                        )

                        # Connect to the index
                        index = pinecone.Index(
                            problem_statement_index_name)

                        # Wait a moment for the index to be fully initialized
                        time.sleep(5)

                    elif database_name == "langchain-ttsh":
                        # Databaes 2: Market Solutions Pinecone.io Database
                        # =======================================================
                        child_pinecone_api_key = os.environ['child_pinecone_api_key']
                        child_pinecone_environment = os.environ['child_pinecone_environment']
                        child_index_name = os.environ['child_index_name']

                        # Initialize connection to pinecone (get API key at app.pinecone.io)
                        pinecone.init(
                            api_key=child_pinecone_api_key,
                            environment=child_pinecone_environment
                        )

                        # Connect to the index
                        index = pinecone.Index(child_index_name)

                        # Wait a moment for the index to be fully initialized
                        time.sleep(5)

                    elif database_name == "market-solutions":
                        # Databaes 2: Market Solutions Pinecone.io Database
                        # =======================================================
                        market_solutions_pinecone_api_key = os.environ[
                            'market_solutions_pinecone_api_key']
                        market_solutions_pinecone_environment = os.environ[
                            'market_solutions_pinecone_environment']
                        market_solutions_index_name = os.environ['market_solutions_index_name']

                        # Initialize connection to pinecone (get API key at app.pinecone.io)
                        pinecone.init(
                            api_key=market_solutions_pinecone_api_key,
                            environment=market_solutions_pinecone_environment
                        )

                        # Connect to the index
                        index = pinecone.Index(market_solutions_index_name)

                        # Wait a moment for the index to be fully initialized
                        time.sleep(5)

                    if index is None:
                        st.error(
                            "There is no Pinecone.io vector database connection.")
                    else:
                        text_field = "text"
                        model_name = 'text-embedding-ada-002'
                        embed = OpenAIEmbeddings(
                            model=model_name,
                            openai_api_key=OPENAI_API_KEY
                        )

                        vectorstore = Pinecone(
                            index, embed.embed_query, text_field
                        )

                        retriever = vectorstore.as_retriever(
                            search_kwargs={'filter': {'source': source}})
                        query = "technology"
                        docs = retriever.get_relevant_documents(query)

                        if len(docs) > 1:
                            st.info(
                                f'At least {len(docs)} documents are found.')
                        elif len(docs) == 1:
                            st.info(f'At least {len(docs)} document is found.')
                        else:
                            st.info(f'No document is found.')

                        response = index.delete(
                            filter={
                                "source": source,
                            }
                        )

                        st.success(
                            f'Vector data/content are deleted from the Pinecone.io "{database_name}"" vector index database.')

except Exception as e:
    error_message = ''
    # st.text('Hello World')
    st.error('An error has occurred. Please try again.', icon="🚨")
    # Just print(e) is cleaner and more likely what you want,
    # but if you insist on printing message specifically whenever possible...
    if hasattr(e, 'message'):
        error_message = e.message
    else:
        error_message = e
    st.error('ERROR MESSAGE: {}'.format(error_message))
    st.error('ERROR MESSAGE: {}'.format(error_message), icon="🚨")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    st.error(f'Error Type: {exc_type}', icon="🚨")
    st.error(f'File Name: {fname}', icon="🚨")
    st.error(f'Line Number: {exc_tb.tb_lineno}', icon="🚨")
    st.error(traceback.format_exc())
