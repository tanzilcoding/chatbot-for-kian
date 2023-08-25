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
import json
import unicodedata
from pathlib import Path

try:
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

    def is_english(c):
        return c.isalpha() and unicodedata.name(c).startswith(('LATIN', 'COMMON'))

    def remove_non_english(lst):
        output = []
        for s in lst:
            filtered = filter(is_english, list(s))
            english_str = ''.join(filtered)
            output.append(english_str)
        return output

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

    st.image('./banner_psms.jpg')

    with st.form("problem-statement-form"):
        st.text_input('Year:', today.year, key="year")
        st.text_area('Background:', key='background',)
        st.text_area('Problem Statement:', key='problem_statement',)
        st.text_area('Desired Outcomes:', key='desired_outcomes',)
        st.text_area('Requestor/Dept/Institution:', key="requestor")
        st.text_area('Funding:', key="funding")
        st.text_area('Contributor:', key="contributor")
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
            contributor = st.session_state.contributor.strip()

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
            elif contributor == "":
                st.error(
                    "Please write the contributor name and try again.")
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
                            # Data to be written
                            dictionary = {
                                "year": year,
                                "background": background,
                                "problem_statement": problem_statement,
                                "desired_outcomes": desired_outcomes,
                                "requestor": requestor,
                                "funding": funding,
                                "contributor": contributor,
                            }

                            # Get the current working directory
                            cwd = os.getcwd()
                            save_folder = f'{cwd}/tmp'

                            # Create a JSON file
                            unique_id = uuid4().hex
                            english_string_list = remove_non_english(
                                [contributor.lower()])
                            english_string = "".join(english_string_list)
                            file_name = f"{english_string}-" + str(unique_id)
                            with open(f"{cwd}/tmp/{file_name}.json", "w") as outfile:
                                json.dump(dictionary, outfile)

                            # Save uploaded file to 'F:/tmp' folder.
                            save_path = Path(
                                save_folder, st.session_state.uploaded_file.name)
                            with open(save_path, mode='wb') as w:
                                w.write(
                                    st.session_state.uploaded_file.getvalue())

                            if save_path.exists():
                                destination_file = f'{cwd}/tmp/{file_name}.pdf'
                                os.rename(save_path, destination_file)
                                save_path = Path(
                                    save_folder, f'{file_name}.pdf')
                                st.info(f'File save path: {save_path}')
                                st.success(
                                    f'File {file_name}.pdf is successfully saved!')
                            else:
                                st.error(
                                    f'File save path: {save_path} does not exist.')

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
