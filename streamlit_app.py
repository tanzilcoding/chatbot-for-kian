

import sys
import os
import time
import uuid
import random
import openai
import pinecone
import streamlit as st
from streamlit_chat import message
import streamlit.components.v1 as components
# from streamlit.components.v1 import html
# import langchain
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains.question_answering import load_qa_chain
# from langchain.chains.qa_with_sources import load_qa_with_sources_chain

# cd /Users/macmini/Documents/Python/Kian-Ping/ChatBotVersion3; streamlit run streamlit_app.py
# https://child-chatgpt.streamlit.app
# https://docs.streamlit.io/library/api-reference/layout/st.empty
# https://towardsdatascience.com/advanced-streamlit-session-state-and-callbacks-for-data-labelling-tool-1e4d9ad32a3f

element_id = 0
placeholder_for_comparison = st.empty
# langchain.verbose = True


def get_source_list(sources):
    chunks = []
    source_list = []
    allowed_file_extensions = ['.pdf']

    try:
        if '\n' in sources:
            chunks = sources.split('\n')
        elif ',' in sources:
            chunks = sources.split(',')
        else:
            source = sources.strip()
            chunks.append(source)
            # source_list.append(source)

        for chunk in chunks:
            chunk = chunk.strip()
            if (len(chunk) >= 5):
                first_two_chars = chunk[:2]
                # print('first_two_chars: {}'.format(first_two_chars))
                if first_two_chars == '- ':
                    chunk = chunk[2:]
                extension = os.path.splitext(chunk)[1]
                extension = extension.lower()
                # print('extension: {}'.format(extension))

                if extension in allowed_file_extensions:
                    # print('{} is in {}'.format(extension, allowed_file_extensions))
                    source_list.append(chunk)
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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        st.error('ERROR MESSAGE: {}'.format(error_message))

    return source_list


def sanitize_answer(raw_answer):
    answer = ''
    chunks = raw_answer.split('\n')
    sub_string_list = []
    sub_string_list.append('- SOURCE:')
    sub_string_list.append('- Source:')
    sub_string_list.append('- source:')
    sub_string_list.append('- SOURCES:')
    sub_string_list.append('- Sources:')
    sub_string_list.append('- sources:')
    sub_string_list.append('(SOURCE:')
    sub_string_list.append('(Source:')
    sub_string_list.append('(source:')
    sub_string_list.append('(SOURCES:')
    sub_string_list.append('(Sources:')
    sub_string_list.append('(sources:')
    sub_string_list.append('SOURCE:')
    sub_string_list.append('Source:')
    sub_string_list.append('source:')
    sub_string_list.append('SOURCES:')
    sub_string_list.append('Sources:')
    sub_string_list.append('sources:')

    try:
        for chunk in chunks:
            temp_string = str(chunk)
            temp_string = temp_string.strip()
            temp_string_lowercase = temp_string.lower()
            answer_text = ''

            if temp_string_lowercase.find('source') != -1:
                for sub_string in sub_string_list:
                    if temp_string.find(sub_string) != -1:
                        # print(f'{sub_string} - {temp_string}')
                        temp_string = temp_string[:temp_string.index(
                            sub_string)]

            # Append answer text
            answer_text = temp_string.strip()
            if len(answer_text) > 0:
                # print(answer_text)
                answer = answer + '\n\n' + answer_text

        answer = answer.strip()
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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        st.error('ERROR MESSAGE: {}'.format(error_message))

    return answer


def accordion(query, sources, answer_type):
    html = ''
    global element_id
    sources = sources.strip()
    source_id = 0
    accordion_height = 0
    # 5 = 290
    # 4 = 240
    # 3 = 190
    # 2 = 140
    # 1 =  90

    # st.sidebar.text(query)

    try:
        if (len(sources)) < 5:
            return html, accordion_height
        else:
            source_list = get_source_list(sources)

            if answer_type == "Chat on projects with sources and summaries":
                model_name = 'text-embedding-ada-002'

                embed = OpenAIEmbeddings(
                    model=model_name,
                    openai_api_key=OPENAI_API_KEY
                )

                text_field = "text"
                index_name = "langchain-ttsh"
                # Switch back to normal index for langchain
                index = pinecone.Index(index_name)

                vectorstore = Pinecone(
                    index, embed.embed_query, text_field
                )

                # Completion llm
                llm = ChatOpenAI(
                    openai_api_key=OPENAI_API_KEY,
                    # model_name='gpt-3.5-turbo',
                    model_name=model,
                    temperature=0.0,
                    # verbose=True
                )

                # st.sidebar.text('Connected to Pinecone....')
                # st.sidebar.text('================================')

        html = html + '<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">'
        html = html + '<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>'
        html = html + '<div class="m-4">'
        html = html + '<div class="accordion" id="myAccordion">'

        for source in source_list:
            source = source.strip()
            source_id = source_id + 1
            element_id = element_id + 1
            # heading_number =

            if answer_type == "Chat on projects with sources":
                summary = ""
            elif answer_type == "Chat on projects with sources and summaries":
                summary_text = ""
                # summary_text = query
                summary = ""

                # xq = llm.encode(query).tolist()
                # result = index.query(xq, top_k=10, filter={'source': source})
                # st.sidebar.text(result)
                # st.sidebar.text('===========================================')

                # Create a retriever
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 3, 'filter': {'source': source}})
                docs = retriever.get_relevant_documents(query)
                # st.sidebar.text(f'Number of retrieved docs = {len(docs)}')
                # st.sidebar.text(docs)
                # st.sidebar.text('===========================================')

                chain = load_summarize_chain(llm, chain_type="stuff")
                # search = vectordb.similarity_search(" ")
                # summary = chain.run(input_documents=search, question="Write a summary within 200 words.")
                summary_text = chain.run(docs)

                summary_text = summary_text.strip()
                if len(summary_text) > 0:
                    # summary = " <br><strong>Summary:</strong> Work in progress for summary"
                    summary = f" <br><strong>Summary:</strong> {summary_text}"
            else:
                summary = ""

            html = html + '<div class="accordion-item">'
            html = html + \
                f'<h2 class="accordion-header" id="heading-{source_id}">'
            html = html + \
                f'<button type="button" class="accordion-button collapsed" data-bs-toggle="collapse" data-bs-target="#collapse-{source_id}">Source {source_id}</button>'
            html = html + '</h2>'
            html = html + \
                f'<div id="collapse-{source_id}" class="accordion-collapse collapse" data-bs-parent="#myAccordion">'
            html = html + '<div class="card-body">'
            html = html + \
                f'<p><strong><a href="https://child.chi.sg/pdffiles/{source}" target="_blank"><span style="overflow-wrap: break-word;">{source}</span></a></strong>{summary}</p>'
            html = html + '</div>'
            html = html + '</div>'
            html = html + '</div>'

        html = html + '</div>'
        html = html + '</div>'

        html = f"""
                {html}
                """

        # html = str(html)
        # accordion_height = 75 + 90 + (source_id - 1) * 50
        # accordion_height = 200 + 90 + (source_id - 1) * 50
        if answer_type == "Chat on projects with sources":
            accordion_height = 150 + 90 + (source_id - 1) * 50
        elif answer_type == "Chat on projects with sources and summaries":
            accordion_height = 400 + 90 + (source_id - 1) * 50
        else:
            accordion_height = 0

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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        st.error('ERROR MESSAGE: {}'.format(error_message))

    return html, accordion_height


def get_last_element_index():
    last_index = -1
    if len(st.session_state.sources) > 0:
        last_index = len(st.session_state.sources) - 1

    return last_index


def compare_two_sources():
    comparison_info = ''

    try:
        last_index = get_last_element_index()
        comparison_type = f"comparison_type_{last_index}"
        comparison_first_source = f"comparison_first_source_{last_index}"
        comparison_second_source = f"comparison_second_source_{last_index}"

        if comparison_type not in st.session_state:
            return
        elif comparison_first_source not in st.session_state:
            return
        elif comparison_second_source not in st.session_state:
            return
        else:
            if st.session_state[comparison_first_source] == "Select One":
                pass
            elif st.session_state[comparison_second_source] == "Select One":
                pass
            elif len(st.session_state.current_source_list) >= 2:
                comparison_title = ''
                comparison_text = ''

                if st.session_state[comparison_type] == "Find Similarities":
                    comparison_title = 'Similarities'
                elif st.session_state[comparison_type] == "Find Differences":
                    comparison_title = 'Differences'

                if st.session_state[comparison_first_source] == st.session_state[comparison_second_source]:
                    comparison_text = f'Oops! You cannot find {comparison_title.lower()} between the same sources.'
                    comparison_info = f'<p style="font-family:sans-serif; font-size: 17px; font-weight: normal;"><strong>{comparison_title} between {st.session_state[comparison_first_source]} and {st.session_state[comparison_second_source]}:</strong> {comparison_text}<p>'
                else:
                    source_label = st.session_state[comparison_first_source]
                    number_list = [int(s)
                                   for s in source_label.split() if s.isdigit()]
                    index = number_list[0] - 1
                    first_source = st.session_state.current_source_list[index]
                    first_source = first_source.strip()

                    source_label = st.session_state[comparison_second_source]
                    number_list = [int(s)
                                   for s in source_label.split() if s.isdigit()]
                    index = number_list[0] - 1
                    second_source = st.session_state.current_source_list[index]
                    second_source = second_source.strip()

                    template = """You will find {comparison_type} between two sources. Answer the question based on the context below. If the question cannot be answered using the information provided, answer with "I don't know".

                    #####Start of {first_source_label}#####
                    {first_source_label}: {first_source_text}
                    #####End of {first_source_label}#####

                    #####Start of {second_source_label}#####
                    {second_source_label}: {second_source_text}
                    #####End of {second_source_label}#####

                    Question: {query}
                    Helpful Answer:"""
                    question = f"What are the {comparison_title.lower()} between {st.session_state[comparison_first_source]} and {st.session_state[comparison_second_source]}?"
                    prompt_template = PromptTemplate(
                        template=template, input_variables=[
                            "comparison_type", "first_source_label", "first_source_text", "second_source_label", "second_source_text", "query"]
                    )

                    first_source_text = ""
                    second_source_text = ""

                    query = st.session_state['past'][last_index]
                    # st.sidebar.text(query)

                    text_field = "text"
                    index_name = "langchain-ttsh"
                    # Switch back to normal index for langchain
                    index = pinecone.Index(index_name)

                    model_name = 'text-embedding-ada-002'
                    embed = OpenAIEmbeddings(
                        model=model_name,
                        openai_api_key=OPENAI_API_KEY
                    )

                    vectorstore = Pinecone(
                        index, embed.embed_query, text_field
                    )

                    # Create a retriever for the first source
                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 5, 'filter': {'source': first_source}})
                    docs = retriever.get_relevant_documents(query)

                    for doc in docs:
                        page_content = str(doc.page_content)
                        page_content = page_content.strip()
                        first_source_text = first_source_text + page_content + "\n\n"

                    first_source_text = first_source_text.strip()

                    # Create a retriever for the second source
                    retriever = vectorstore.as_retriever(
                        search_kwargs={"k": 5, 'filter': {'source': second_source}})
                    docs = retriever.get_relevant_documents(query)

                    for doc in docs:
                        page_content = str(doc.page_content)
                        page_content = page_content.strip()
                        second_source_text = second_source_text + page_content + "\n\n"

                    second_source_text = second_source_text.strip()

                    # initialize the models
                    # model = "gpt-3.5-turbo"
                    model = "gpt-3.5-turbo"
                    openai = OpenAI(
                        model_name=model,
                        openai_api_key=OPENAI_API_KEY,
                    )

                    comparison_text = openai(
                        prompt_template.format(
                            comparison_type=comparison_title.lower(),
                            first_source_label=st.session_state[comparison_first_source],
                            first_source_text=first_source_text,
                            second_source_label=st.session_state[comparison_second_source],
                            second_source_text=second_source_text,
                            query=f"What are the {comparison_title.lower()} between {st.session_state[comparison_first_source]} and {st.session_state[comparison_second_source]}?",
                        )
                    )

                    # comparison_text = f'Information is coming soon... - ' + question
                    comparison_info = comparison_info + \
                        f'<p style="font-family:sans-serif; font-size: 17px; font-weight: normal;"><strong>{comparison_title} between {st.session_state[comparison_first_source]} and {st.session_state[comparison_second_source]}:</strong><p>'
                    comparison_info = comparison_info + \
                        f'<p style="font-family:sans-serif; font-size: 17px; font-weight: normal; overflow-wrap: break-word; font-style: italic;">{st.session_state[comparison_first_source]}: {first_source}<p>'
                    comparison_info = comparison_info + \
                        f'<p style="font-family:sans-serif; font-size: 17px; font-weight: normal; overflow-wrap: break-word; font-style: italic;">{st.session_state[comparison_second_source]}: {second_source}<p>'
                    comparison_info = comparison_info + \
                        f'<p style="font-family:sans-serif; font-size: 17px; font-weight: normal;"><strong>{comparison_title}:</strong> {comparison_text} <p>'

            # st.sidebar.text(comparison_info)
            # Update the last value
            if len(st.session_state.sources) > 0:
                last_index = len(st.session_state.sources) - 1
                # st.sidebar.text(f'last_index: {last_index}')

                if last_index >= 0:
                    st.session_state.source_comparison_list[last_index] = comparison_info
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
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        st.error('ERROR MESSAGE: {}'.format(error_message))


try:
    st.set_page_config(page_title="CHILD Project Collection",
                       page_icon=":robot_face:")
    # system message to 'prime' the model
    primer = f"""You are Q&A bot. A highly intelligent system that answers
                user questions based on the information provided by the user above
                each question. If the information can not be found in the information
                provided by the user you truthfully say "I don't know".
                """

    # Set environment variables
    pinecone_api_key = os.environ['pinecone_api_key']
    pinecone_environment = os.environ['pinecone_environment']
    openai.organization = os.environ['openai_organization']
    openai.api_key = os.environ['openai_api_key']
    OPENAI_API_KEY = os.environ['openai_api_key']

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment  # find next to API key in console
    )

    # index_name = "langchain-pdf"
    index_name = "langchain-ttsh"
    embed_model = "text-embedding-ada-002"
    # connect to index
    #index = pinecone.GRPCIndex(index_name)
    index = pinecone.Index(index_name)
    # wait a moment for the index to be fully initialized
    time.sleep(1)

    # Setting page title and header
    # st.set_page_config(page_title="CHILD ChatGPT", page_icon=":robot_face:")

    st.write("[Home](https://child.chi.sg) / Chat")

    st.markdown("<h1 style='text-align: center;'>CHILD Project Collection</h1>",
                unsafe_allow_html=True)
    st.image('./banner2.jpg')

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": primer}
        ]
    if 'sources' not in st.session_state:
        st.session_state['sources'] = []
    if 'current_source_list' not in st.session_state:
        st.session_state['current_source_list'] = []
    if 'source_comparison_list' not in st.session_state:
        st.session_state['source_comparison_list'] = []

    if 'comparison_type' not in st.session_state:
        st.session_state['comparison_type'] = ''
    if 'comparison_type_first_source' not in st.session_state:
        st.session_state['comparison_type_first_source'] = ''
    if 'comparison_type_second_source' not in st.session_state:
        st.session_state['comparison_type_second_source'] = ''

    if 'accordion_html_code' not in st.session_state:
        st.session_state['accordion_html_code'] = []
    if 'accordion_height' not in st.session_state:
        st.session_state['accordion_height'] = []
    # if 'model_name' not in st.session_state:
    #     st.session_state['model_name'] = []
    # if 'cost' not in st.session_state:
    #     st.session_state['cost'] = []
    # if 'total_tokens' not in st.session_state:
    #     st.session_state['total_tokens'] = []
    # if 'total_cost' not in st.session_state:
    #     st.session_state['total_cost'] = 0.0

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    # st.sidebar.title("Sidebar")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['sources'] = []
        st.session_state['current_source_list'] = []
        st.session_state['source_comparison_list'] = []
        st.session_state['comparison_type'] = ''
        st.session_state['comparison_type_first_source'] = ''
        st.session_state['comparison_type_second_source'] = ''
        st.session_state['accordion_html_code'] = []
        st.session_state['accordion_height'] = []
        # st.session_state['number_tokens'] = []
        # st.session_state['model_name'] = []
        # st.session_state['cost'] = []
        # st.session_state['total_cost'] = 0.0
        # st.session_state['total_tokens'] = []
        # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

    st.sidebar.image("./child logo_with brushstroke3.png",
                     use_column_width=True)

    with st.sidebar:
        st.caption("Developed from open source codes for a search based on CHILD collection of projects. Do be aware that there will be longer waiting time as the chatbot search through our knowledge base for chat on projects with sources and summaries")

    answer_type = st.sidebar.radio(
        "Choose your preference:", ("Chat only", "Chat on projects with sources", "Chat on projects with sources and summaries"))
    # model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    model_name = "GPT-3.5"

    # Comparison tool
    # Clear the placeholder
    placeholder_for_comparison_type = st.sidebar.empty()
    placeholder_for_source_1 = st.sidebar.empty()
    placeholder_for_source_2 = st.sidebar.empty()

    if answer_type == "Chat only":
        pass
    else:
        if len(st.session_state.current_source_list) >= 2:
            last_index = get_last_element_index()
            comparison_type_key = f"comparison_type_{last_index}"

            # Add a comparison type to the sidebar
            comparison_type = placeholder_for_comparison_type.radio(
                "Compare 2 (two) sources",
                ["None", "Find Similarities", "Find Differences"],
                key=comparison_type_key,
                on_change=compare_two_sources,
            )

            if comparison_type == "None":
                pass
            else:
                dropdown_sources = ['Select One']
                source_counter = 0
                for source in st.session_state.current_source_list:
                    source_counter = source_counter + 1
                    source_label = f'Source {source_counter}'
                    dropdown_sources.append(source_label)

                # Add sources to to Source # 1
                placeholder_for_source_1.selectbox(
                    "Select the first source",
                    dropdown_sources,
                    key=f"comparison_first_source_{last_index}",
                    on_change=compare_two_sources,
                )

                dropdown_sources = ['Select One']
                source_counter = 0
                for source in st.session_state.current_source_list:
                    source_counter = source_counter + 1
                    source_label = f'Source {source_counter}'
                    dropdown_sources.append(source_label)

                # Add sources to to Source # 2
                placeholder_for_source_2.selectbox(
                    "Select the second source",
                    dropdown_sources,
                    key=f"comparison_second_source_{last_index}",
                    on_change=compare_two_sources,
                )

    counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    # st.sidebar.text("")
    # st.sidebar.text("")
    # st.sidebar.text("")
    # st.sidebar.text("Do you want to reset?")
    # clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo"
    else:
        model = "gpt-4"

    # Generate a response
    def generate_response(query):
        # query = query + 'Give me a detailed answer.'
        # query = query + ' I do not need information references. Give me as detailed answer as possible. Answer based on only the information provided.'
        response = ''
        sources = ''
        model_name = 'text-embedding-ada-002'

        embed = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=OPENAI_API_KEY
        )

        text_field = "text"
        # Switch back to normal index for langchain
        index = pinecone.Index(index_name)

        vectorstore = Pinecone(
            index, embed.embed_query, text_field
        )

        # completion llm
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            # model_name='gpt-3.5-turbo',
            model_name=model,
            temperature=0.0,
            # verbose=True
        )

        messages = [
            SystemMessage(
                content=primer
            ),
            HumanMessage(
                content=query
                # content="Please give me a detailed answer to my question."
            ),
        ]

        llm(messages)

        if answer_type == "Chat only":
            docs = vectorstore.similarity_search(
                query,  # our search query
                k=5  # return 5 most relevant docs
            )

            chain = load_qa_chain(llm, chain_type="stuff",
                                  # verbose=True
                                  )

            question = query
            raw_answer = chain.run(input_documents=docs, question=question)

            # st.sidebar.text(raw_answer)

            raw_answer = raw_answer.strip()
            response = sanitize_answer(raw_answer)
        else:
            qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 4}),
                # verbose=True,
                # return_source_documents=True,
                # chain_type_kwargs={
                #    "verbose": True,
                # }
            )

            question = query
            result = qa_with_sources(question)

            raw_answer = result['answer']
            response = sanitize_answer(raw_answer)
            sources = result['sources']

            st.session_state['messages'].append(
                {"role": "assistant", "content": response})

        # print(st.session_state['messages'])
        # total_tokens = completion.usage.total_tokens
        # prompt_tokens = completion.usage.prompt_tokens
        # completion_tokens = completion.usage.completion_tokens
        # return response, total_tokens, prompt_tokens, completion_tokens
        return response, sources

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=50)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            # The user has pressed the submitted button
            st.session_state.current_source_list = []
            # Clear the placeholders
            placeholder_for_comparison_type.empty()
            placeholder_for_source_1.empty()
            placeholder_for_source_2.empty()

            # output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
            output, sources = generate_response(user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['sources'].append(sources)

            st.session_state['source_comparison_list'].append('')

            accordion_html_code = ""
            accordion_height = 0

            if answer_type == "Chat only":
                pass
            else:
                query = user_input
                sources = sources

                sources = sources.strip()
                # st.sidebar.text(f'sources: {sources}')
                if (len(sources)) >= 5:
                    # st.write(f"Sources: {st.session_state['sources'][i]}")
                    accordion_html_code, accordion_height = accordion(
                        query, sources, answer_type)
                    accordion_html_code = str(accordion_html_code)

            st.session_state['accordion_html_code'].append(accordion_html_code)
            st.session_state['accordion_height'].append(accordion_height)
            # st.session_state['model_name'].append(model_name)
            # st.session_state['total_tokens'].append(total_tokens)

            # from https://openai.com/pricing#language-models
            # if model_name == "GPT-3.5":
            #     cost = total_tokens * 0.002 / 1000
            # else:
            #     cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

            # st.session_state['cost'].append(cost)
            # st.session_state['total_cost'] += cost

            # source_list = get_source_list(sources)
            st.session_state.current_source_list = get_source_list(sources)

            if len(st.session_state.current_source_list) >= 2:
                last_index = get_last_element_index()

                # Add a comparison type to the sidebar
                placeholder_for_comparison_type.radio(
                    "Compare 2 (two) sources",
                    ["None", "Find Similarities", "Find Differences"],
                    key=f"comparison_type_{last_index}",
                    on_change=compare_two_sources,
                )

                dropdown_sources = ['Select One']
                source_counter = 0
                for source in st.session_state.current_source_list:
                    source_counter = source_counter + 1
                    source_label = f'Source {source_counter}'
                    dropdown_sources.append(source_label)

                # Add sources to to Source # 1
                placeholder_for_source_1.selectbox(
                    "Select the first source",
                    dropdown_sources,
                    key=f"comparison_first_source_{last_index}",
                    on_change=compare_two_sources,
                )

                dropdown_sources = ['Select One']
                source_counter = 0
                for source in st.session_state.current_source_list:
                    source_counter = source_counter + 1
                    source_label = f'Source {source_counter}'
                    dropdown_sources.append(source_label)

                # Add sources to to Source # 2
                placeholder_for_source_2.selectbox(
                    "Select the second source",
                    dropdown_sources,
                    key=f"comparison_second_source_{last_index}",
                    on_change=compare_two_sources,
                )

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user=True, key=str(i) + '_user')
                message(st.session_state["generated"][i], key=str(i))

                accordion_html_code = st.session_state["accordion_html_code"][i]
                accordion_height = st.session_state["accordion_height"][i]
                sources = st.session_state["sources"][i]

                if accordion_height > 0:
                    components.html(
                        accordion_html_code,
                        height=accordion_height,
                    )

                if i < len(st.session_state.source_comparison_list):
                    source_comparison_html_code = st.session_state.source_comparison_list[i]
                    source_comparison_html_code = str(
                        source_comparison_html_code)
                    source_comparison_html_code = source_comparison_html_code.strip()

                    if source_comparison_html_code != "":
                        st.markdown(source_comparison_html_code,
                                    unsafe_allow_html=True)

                        # components.html(
                        #     source_comparison_html_code,
                        #     height=100,
                        # )
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
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    st.error('ERROR MESSAGE: {}'.format(error_message))
    more_info = f'{exc_type} ----- {fname} ----- {exc_tb.tb_lineno}'
    st.error('MORE INFO: {}'.format(more_info))
