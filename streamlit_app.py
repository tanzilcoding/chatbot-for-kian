import streamlit as st
import pinecone
import os
import datetime
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
# from langchain.document_loaders import UnstructuredFileLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from typing_extensions import Concatenate
from uuid import uuid4
from tqdm.auto import tqdm
import random
from tqdm.auto import tqdm
from time import sleep
import streamlit.components.v1 as components
from streamlit_js_eval import streamlit_js_eval
from streamlit.components.v1 import html

try:
    import environment_variables
except ImportError:
    pass

try:
    # Set environment variables
    # Set org ID and API key
    # openai.organization = "<YOUR_OPENAI_ORG_ID>"
    # openai.organization = os.environ['openai_organization']

    os.environ['openai_api_key'] = "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"
    os.environ['pinecone_api_key'] = "4907cba7-f52a-4211-b521-e5b6ce57db17"
    os.environ['pinecone_environment'] = "us-west1-gcp-free"
    os.environ['index_name'] = "langchain"
    # =======================================================
    OPENAI_API_KEY = os.environ['openai_api_key']
    pinecone_api_key = os.environ['pinecone_api_key']
    pinecone_environment = os.environ['pinecone_environment']
    openai.api_key = OPENAI_API_KEY
    index_name = os.environ['index_name']
    # ==================================================== #
    today = datetime.date.today()

    def make_safe_filename(s):
        def safe_char(c):
            if c.isalnum():
                return c
            else:
                return "_"
        return "".join(safe_char(c) for c in s).rstrip("_")

    def get_correct_file_name(file_name):
        extension = os.path.splitext(file_name)[1]
        file_name = file_name.replace(extension, "")

        # Clean it in one fell swoop.
        new_file_name = make_safe_filename(file_name)
        new_file_name = new_file_name.replace("__", "_")
        new_file_name = new_file_name.replace("__", "_")
        new_file_name = new_file_name.replace("__", "_")
        new_file_name = new_file_name.replace("__", "_")

        first_char = new_file_name[0]
        if (first_char == '_'):
            new_file_name = new_file_name[1:]

        file_name = file_name + extension
        correct_file_name = new_file_name + extension

        return correct_file_name

    tokenizer_name = tiktoken.encoding_for_model('gpt-4')
    # print(tokenizer_name.name)
    tokenizer = tiktoken.get_encoding(tokenizer_name.name)

    # create the length function
    def tiktoken_len(text):
        tokens = tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    page_title = "Problem Statement Uploader"
    st.set_page_config(page_title=page_title,)
    st.title(page_title)

    st.markdown('<a href="https://psms-matchmaker.streamlit.app">Search PS to Match Make</a>', unsafe_allow_html=True)

    with st.form("problem-statement-form"):
        st.text_input('Year:', today.year, key="year")
        st.text_area('Background:', key='background',)
        st.text_area('Problem Statement:', key='problem_statement',)
        st.text_area('Desired Outcomes:', key='desired_outcomes',)
        st.text_area('Requestor/Dept/Institution:', key="requestor")
        st.text_area('Funding:', key="funding")
        st.file_uploader(
            "Upload a PDF file for the problem statement:", type="pdf", key="uploaded_file")

        is_submitted = st.form_submit_button(
            label="Submit", )

        if is_submitted:
            year = st.session_state.year.strip()
            background = st.session_state.background.strip()
            problem_statement = st.session_state.problem_statement.strip()
            desired_outcomes = st.session_state.desired_outcomes.strip()
            requestor = st.session_state.requestor.strip()
            funding = st.session_state.funding.strip()

            if year == "":
                st.error("Please write a year and try again.")
            elif year.isdigit() == False:
                st.error(
                    "Please write a number for the year and try again.")
            elif int(year) < 2023:
                st.error(
                    "Please write a year greater than or equal to 2023 and try again.")
            elif int(year) > 2030:
                st.error(
                    "Please write a year smaller than or equal to 2030 and try again.")
            elif background == "":
                st.error("Please write a background and try again.")
            elif problem_statement == "":
                st.error("Please write a problem statement and try again.")
            elif desired_outcomes == "":
                st.error("Please write desired outcomes and try again.")
            elif requestor == "":
                st.error(
                    "Please write a Requestor/Dept/Institution and try again.")
            elif funding == "":
                st.error(
                    "Please write funding information and try again.")
            elif st.session_state.uploaded_file is None:
                st.error(
                    "Please upload a PDF file for the problem statement and try again.")
            else:
                # print(f'uploaded_file.name: {uploaded_file.name}')
                file_name, file_extension = os.path.splitext(
                    st.session_state.uploaded_file.name)

                if file_extension == ".pdf":
                    correct_file_name = get_correct_file_name(
                        st.session_state.uploaded_file.name)

                    if correct_file_name != st.session_state.uploaded_file.name:
                        st.error(
                            f'Your file name is: {st.session_state.uploaded_file.name}')
                        st.error(
                            f'The correct file name should be: {correct_file_name}')
                    else:
                        st.info(
                            "REMINDER: Beware! Is your data/information correct? Please double check. The reason is - there is no way to find out these data that are saved only from this submission. It is recommended you check your data right now. This is your last chance to check and submit correct data.\n\n\n1. Is the YEAR correct?\n\n\n2. Is the BACKGROUND correct?\n\n\n3. Is the PROBLEM STATEMENT correct?\n\n\n4. Are the DESIRED OUTCOMES correct?\n\n\n5. Is the REQUESTOR/DEPT/INSTITUTION information correct?\n\n\n6. Is the FUNDING information correct?\n\n\n7. Did you select the correct PDF file?\n\n\nIf you are 100% sure, please click on the following button.")

                        st.info(
                            'If you are 100% sure that the provided information is correct, please check the following checkbox and press the "Submit" button again.')
                        in_information_correct = st.checkbox(
                            'I guarantee that the information is correct. Add to the database.')

                        if in_information_correct:
                            pdf_reader = PdfReader(
                                st.session_state.uploaded_file)
                            text = ""
                            docs = []
                            for page in pdf_reader.pages:
                                text = page.extract_text()
                                doc = Document(page_content=text,
                                               metadata={"source": st.session_state.uploaded_file.name})
                                docs.append(doc)

                            upserted_count = 0

                            if len(docs) > 0:
                                # st.sidebar.text(docs)
                                # st.sidebar.text(docs[0])
                                text_splitter = RecursiveCharacterTextSplitter(
                                    chunk_size=500,
                                    chunk_overlap=20,
                                    length_function=tiktoken_len,
                                    separators=["\n\n", "\n", " ", ""]
                                )

                                if len(tokenizer_name.name) > 0:
                                    # st.info('{} is the name of your tokenizer.'.format(tokenizer_name.name))
                                    chunks = []
                                    for idx, page in enumerate(tqdm(docs)):
                                        content = page.page_content
                                        if len(content) > 100:
                                            texts = text_splitter.split_text(
                                                content)
                                            num = random.random()
                                            chunks.extend([{
                                                'id': str(uuid4()),
                                                'text': texts[i],
                                                'chunk': i,
                                                'source': st.session_state.uploaded_file.name,
                                                'year': year,
                                                'background': background,
                                                'problem_statement': problem_statement,
                                                'desired_outcomes': desired_outcomes,
                                                'requestor': requestor,
                                                'funding': funding,
                                            } for i in range(len(texts))])

                                    if len(chunks) > 0:
                                        if len(chunks) > 1:
                                            st.info('{} text chunks are ready to be converted into vectors.'.format(
                                                len(chunks)))
                                        else:
                                            st.info('{} text chunk is ready to be converted into vectors.'.format(
                                                len(chunks)))

                                        # initialize connection to pinecone (get API key at app.pinecone.io)
                                        pinecone.init(
                                            api_key=pinecone_api_key,
                                            environment=pinecone_environment  # find next to API key in console
                                        )
                                        index = pinecone.Index(index_name)
                                        sleep(5)

                                        # stats = index.describe_index_stats()
                                        # last_total_vector_count = stats['total_vector_count']
                                        # last_total_vector_count = int(
                                        #     last_total_vector_count)
                                        # st.info(
                                        #     f'Last total vector count: {last_total_vector_count}')

                                        batch_size = 100  # how many embeddings we create and insert at once
                                        embed_model = "text-embedding-ada-002"

                                        for i in tqdm(range(0, len(chunks), batch_size)):
                                            # find end of batch
                                            i_end = min(
                                                len(chunks), i+batch_size)
                                            meta_batch = chunks[i:i_end]
                                            # get ids
                                            ids_batch = [x['id']
                                                         for x in meta_batch]
                                            # get texts to encode
                                            texts = [x['text']
                                                     for x in meta_batch]
                                            # create embeddings (try-except added to avoid RateLimitError)
                                            try:
                                                res = openai.Embedding.create(
                                                    input=texts, engine=embed_model)
                                            except:
                                                done = False
                                                while not done:
                                                    sleep(5)
                                                    try:
                                                        res = openai.Embedding.create(
                                                            input=texts, engine=embed_model)
                                                        done = True
                                                    except:
                                                        pass
                                            embeds = [record['embedding']
                                                      for record in res['data']]

                                            # cleanup metadata
                                            meta_batch = [{
                                                'text': x['text'],
                                                'chunk': x['chunk'],
                                                'source': os.path.basename(x['source']),
                                                'year': x['year'],
                                                'background': x['background'],
                                                'problem_statement': x['problem_statement'],
                                                'desired_outcomes': x['desired_outcomes'],
                                                'requestor': x['requestor'],
                                                'funding': x['funding'],
                                            } for x in meta_batch]
                                            to_upsert = list(
                                                zip(ids_batch, embeds, meta_batch))

                                            # upsert to Pinecone
                                            upsert_response = index.upsert(
                                                vectors=to_upsert)
                                            upserted_count = upserted_count + \
                                                upsert_response['upserted_count']

                                        # st.sidebar.text(stats)

                                        if upserted_count > 0:
                                            st.success(
                                                f'Total upsert count: {upserted_count}')
                                            # st.info('This page will redirect you to the "Search PS to Match Make" pagein the next 5 (five) seconds.')

                                            if len(chunks) > 1:
                                                st.success('{} text chunks are converted into vectors and uploaded to your Pinecone index: {}.'.format(
                                                    len(chunks), index_name))
                                            else:
                                                st.success('{} text chunk is converted into vectors and uploaded to your Pinecone index: {}.'.format(
                                                    len(chunks), index_name))

                                            # Reload the page
                                            # sleep(5)
                                            # Simulate an HTTP redirect:
                                            # streamlit_js_eval(js_expressions='window.location.replace("https://psms-matchmaker.streamlit.app");')
                                            # streamlit_js_eval(js_expressions='window.location.href = "https://psms-matchmaker.streamlit.app"')
                                            # streamlit_js_eval(js_expressions="parent.window.location.reload()")
                                            # nav_to("https://psms-matchmaker.streamlit.app")
                                            st.markdown('<a href="https://psms-matchmaker.streamlit.app" target="_self">Search PS to Match Make</a>', unsafe_allow_html=True)
                                        else:
                                            st.error(
                                                f'Oops! Upsert operation was unsuccessful. This means there was no data uploaded to the Pinecone.io vector index database. So, there was no change. Please try again.')

                                        # st.session_state.problem_statement_text = ''
                                    else:
                                        print('Oops! No text chunk was created.')
                                else:
                                    print('Oops! No tokenizer could be created.')
                            else:
                                print(
                                    'The script does not find any LangChain documents from your PDF files.')
                                print(
                                    'Did you really upload any valid input files?')
                else:
                    st.error(
                        "Only PDF Files (.pdf extension) are accepted. Please try again.")
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
