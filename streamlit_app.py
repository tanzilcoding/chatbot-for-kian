import os
import sys
import time
import traceback
import openai
import pinecone
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain

try:
    import environment_variables
except ImportError:
    pass

try:
    # Setting page title and header
    st.set_page_config(page_title="AI ChatBot", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>AI ChatBot 😬</h1>",
                unsafe_allow_html=True)

    # Step 1: Get common environment variables
    OPENAI_API_KEY = os.environ['openai_api_key']
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    text_field = "text"
    model_name = 'text-embedding-ada-002'
    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    # Step 2: Get Pinecone.io database specific environment variables

    # Databaes 1: Problem Statement Pinecone.io Database
    # =======================================================
    problem_statement_pinecone_api_key = os.environ['problem_statement_pinecone_api_key']
    problem_statement_pinecone_environment = os.environ['problem_statement_pinecone_environment']
    problem_statement_index_name = os.environ['problem_statement_index_name']

    # Initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=problem_statement_pinecone_api_key,
        environment=problem_statement_pinecone_environment
    )

    # Connect to the index
    problem_statement_index = pinecone.Index(problem_statement_index_name)
    # Wait a moment for the index to be fully initialized
    time.sleep(1)

    problem_statement_vectorstore = Pinecone(
        problem_statement_index, embed.embed_query, text_field
    )
    # ==================================================== #

    # Databaes 2: CHILD Pinecone.io Database
    # =======================================================
    child_pinecone_api_key = os.environ['child_pinecone_api_key']
    child_pinecone_environment = os.environ['child_pinecone_environment']
    child_index_name = os.environ['child_index_name']

    # Initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=child_pinecone_api_key,
        environment=child_pinecone_environment  # find next to API key in console
    )

    # Connect to the index
    child_index = pinecone.Index(child_index_name)
    # Wait a moment for the index to be fully initialized
    time.sleep(1)

    child_vectorstore = Pinecone(
        child_index, embed.embed_query, text_field
    )
    # ==================================================== #

    # Databaes 3: Market Solutions Pinecone.io Database
    # =======================================================
    market_solutions_pinecone_api_key = os.environ['market_solutions_pinecone_api_key']
    market_solutions_pinecone_environment = os.environ['market_solutions_pinecone_environment']
    market_solutions_index_name = os.environ['market_solutions_index_name']

    # Initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=market_solutions_pinecone_api_key,
        environment=market_solutions_pinecone_environment
    )

    # Connect to the index
    market_solutions_index = pinecone.Index(market_solutions_index_name)
    # Wait a moment for the index to be fully initialized
    time.sleep(1)

    market_solutions_vectorstore = Pinecone(
        market_solutions_index, embed.embed_query, text_field
    )
    # ==================================================== #

    # Initialise session state variables
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
    if 'model_name' not in st.session_state:
        st.session_state['model_name'] = []
    if 'problem_statement_list' not in st.session_state:
        st.session_state['problem_statement_list'] = []
    if 'child_response' not in st.session_state:
        st.session_state['child_response'] = []

    # Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
    st.sidebar.title("Sidebar")
    model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    counter_placeholder = st.sidebar.empty()
    # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    # Map model names to OpenAI model IDs
    if model_name == "GPT-3.5":
        model = "gpt-3.5-turbo-16k"
    else:
        model = "gpt-4"

    # Initialize the large language model
    llm = ChatOpenAI(
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
        model_name=model,
    )

    # reset everything
    if clear_button:
        st.session_state['generated'] = []
        st.session_state['past'] = []
        st.session_state['messages'] = [
            {"role": "system", "content": "You are a helpful assistant."}
        ]
        st.session_state['model_name'] = []
        # counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")

    # generate a response

    def generate_response(prompt):
        query = prompt
        st.session_state['messages'].append(
            {"role": "user", "content": prompt})

        ######################################################
        # docs = vectorstore.similarity_search(
        #     query,  # our search query
        #     k=3,  # return 3 most relevant docs
        #     # include_metadata=True
        # )

        # for doc in docs:
        #     st.sidebar.text(doc.metadata['problem_statement'])

        docs_and_scores = problem_statement_vectorstore.similarity_search_with_score(
            query)

        raw_problem_statement_list = []
        problem_statement_list = []
        for doc in docs_and_scores:
            # st.sidebar.text(doc.metadata['problem_statement'])
            year = list(doc)[0].metadata['year']
            requestor = list(doc)[0].metadata['requestor']
            problem_statement = list(doc)[0].metadata['problem_statement']
            raw_problem_statement = problem_statement.strip()
            raw_problem_statement = raw_problem_statement.lower()
            raw_problem_statement = raw_problem_statement.replace(" ", "")
            raw_problem_statement = raw_problem_statement.replace("\n", "")

            if raw_problem_statement not in raw_problem_statement_list:
                raw_problem_statement_list.append(raw_problem_statement)

                # st.text(problem_statement)
                # st.sidebar.text(doc)
                score = list(doc)[1]
                score = float(score)
                score = score * 100
                # st.text(score)

                if score >= 80:
                    score = str(round(score, 2))
                    problem_statement_list.append(
                        {"score": score, "problem_statement": problem_statement, "year": year, "requestor": requestor, })

        max_token_limit = 4096
        # Problem Statement
        ######################################################
        problem_statement_memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=max_token_limit,
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=problem_statement_vectorstore.as_retriever(),
            # retriever=problem_statement_vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=problem_statement_memory,
        )
        response = chain.run({'question': query})
        problem_statement_response = response.strip()
        ######################################################

        # Problem Statement
        ######################################################
        child_memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=max_token_limit,
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=child_vectorstore.as_retriever(),
            # retriever=problem_statement_vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=child_memory,
        )
        response = chain.run({'question': query})
        child_response = response.strip()
        ######################################################

        st.session_state['messages'].append(
            {"role": "assistant", "content": problem_statement})

        return problem_statement_response, problem_statement_list, child_response

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            problem_statement_response, problem_statement_list, child_response = generate_response(
                user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(problem_statement_response)
            st.session_state['model_name'].append(model_name)
            st.session_state['problem_statement_list'].append(
                problem_statement_list)
            st.session_state['child_response'].append(child_response)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i],
                        is_user=True, key=str(i) + '_user')

                if len(st.session_state["problem_statement_list"][i]) < 1:
                    st.markdown(
                        f"""<span style="word-wrap:break-word;"><strong>Oops!</strong> No matching problem statement is found.</span>""", unsafe_allow_html=True)
                elif len(st.session_state["problem_statement_list"][i]) == 1:
                    for problem_statement_data in st.session_state["problem_statement_list"][i]:
                        score = problem_statement_data["score"]
                        year = problem_statement_data["year"]
                        requestor = problem_statement_data["requestor"]
                        requestor = requestor.strip()
                        requestor = requestor.replace('\n', '<br>')
                        problem_statement = problem_statement_data["problem_statement"]
                        st.markdown(
                            f"""<span style="word-wrap:break-word;"><strong>Problem Statement Found:</strong> {problem_statement}</span> <span style="word-wrap:break-word; font-style: italic;">(Relevance Score: {score}%)</span>""", unsafe_allow_html=True)
                        st.markdown(
                            f"""<span style="word-wrap:break-word;"><strong>Year:</strong> {year}""", unsafe_allow_html=True)
                        st.markdown(
                            f"""<span style="word-wrap:break-word;"><strong>Requestor/Dept/Institution:</strong><br>{requestor}""", unsafe_allow_html=True)
                else:
                    counter = 0
                    for problem_statement_data in st.session_state["problem_statement_list"][i]:
                        counter = counter + 1
                        score = problem_statement_data["score"]
                        year = problem_statement_data["year"]
                        requestor = problem_statement_data["requestor"]
                        requestor = requestor.strip()
                        requestor = requestor.replace('\n', '<br>')
                        problem_statement = problem_statement_data["problem_statement"]
                        st.markdown(
                            f"""<span style="word-wrap:break-word;"><strong>Problem Statement Found {counter}:</strong> {problem_statement}</span> <span style="word-wrap:break-word; font-style: italic;">(Relevance Score: {score}%)</span>""", unsafe_allow_html=True)
                        st.markdown(
                            f"""<span style="word-wrap:break-word;"><strong>Year:</strong> {year}""", unsafe_allow_html=True)
                        st.markdown(
                            f"""<span style="word-wrap:break-word;"><strong>Requestor/Dept/Institution:</strong><br>{requestor}""", unsafe_allow_html=True)

                generated_answer = st.session_state["generated"][i]
                st.markdown(
                    f"""<span style="word-wrap:break-word;"><strong>Summary:</strong><br>{generated_answer}""", unsafe_allow_html=True)
                message(
                    f'Answer from the CHILD Database: {st.session_state["child_response"][i]}', key=str(i))


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
    st.error('ERROR MESSAGE: {}'.format(error_message), icon="🚨")
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    # st.error(f'Error Type: {exc_type}', icon="🚨")
    # st.error(f'File Name: {fname}', icon="🚨")
    # st.error(f'Line Number: {exc_tb.tb_lineno}', icon="🚨")
    # print(traceback.format_exc())
