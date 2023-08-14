import os
import time
import openai
import pinecone
import streamlit as st
from streamlit_chat import message
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import HumanMessage, SystemMessage, AIMessage

try:
    import environment_variables
except ImportError:
    pass

try:
    # Setting page title and header
    st.set_page_config(page_title="AI ChatBot", page_icon=":robot_face:")
    st.markdown("<h1 style='text-align: center;'>AI ChatBot 😬</h1>",
                unsafe_allow_html=True)

    # Get environment variables
    # openai.organization = os.environ['openai_organization']
    # =======================================================
    OPENAI_API_KEY = os.environ['openai_api_key']
    pinecone_api_key = os.environ['pinecone_api_key']
    pinecone_environment = os.environ['pinecone_environment']
    openai.api_key = OPENAI_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    index_name = os.environ['index_name']
    # ==================================================== #

    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=pinecone_api_key,
        environment=pinecone_environment  # find next to API key in console
    )

    # connect to index
    index = pinecone.Index(index_name)
    # wait a moment for the index to be fully initialized
    time.sleep(1)
    # stats = index.describe_index_stats()

    # get openai api key from platform.openai.com
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or 'OPENAI_API_KEY'

    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    text_field = "text"

    # switch back to normal index for langchain
    index = pinecone.Index(index_name)

    vectorstore = Pinecone(
        index, embed.embed_query, text_field
    )

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

        docs_and_scores = vectorstore.similarity_search_with_score(query)

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

        # completion llm
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=model,
            temperature=0.0
        )

        # system message to 'prime' the model
        primer = f"""You are a Q&A bot. A highly intelligent system that answers
                user questions based on the information provided by the user above
                each question. If the information can not be found in the information
                provided by the user you truthfully say "I don't know".
                """

        messages = [
            SystemMessage(
                content=primer
            ),
            HumanMessage(
                content=query
            ),
        ]

        llm(messages)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        response = qa.run(query)
        response = response.strip()
        ######################################################
        # response = ""
        st.session_state['messages'].append(
            {"role": "assistant", "content": response})

        return response, problem_statement_list

    # container for chat history
    response_container = st.container()
    # container for text box
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area("You:", key='input', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            output, problem_statement_list = generate_response(
                user_input)
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
            st.session_state['model_name'].append(model_name)
            st.session_state['problem_statement_list'].append(
                problem_statement_list)

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

                message(
                    f'Summary: {st.session_state["generated"][i]}', key=str(i))


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
