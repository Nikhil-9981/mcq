import os
import json
import traceback
import pandas as pd 
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

from langchain.callbacks import get_openai_callback



load_dotenv()

key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(openai_api_key = key,model_name= "gpt-3.5-turbo", temperature = 0.5)

template = """
Text: {text}
You are an expert MCQ maaker. Given the above text, it is our job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone.
Make sure to questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and it as a guise. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}
"""


quiz_generation_prompt = PromptTemplate(
    input_variables = ["text","number","subject","tone","response_json"],
    template= template
)

quiz_chain = LLMChain(llm=llm, prompt = quiz_generation_prompt, output_key="quiz",verbose=True)

template2 = """
You are an expert english grammarian and writer. Given a Multiple choice quiz for {subject} stundents. \
You need to evaluate the complexity of the queston and give a complete analysis of the quiz. Only use at max 50 words for complexity.
If the quiz questions is not as per with the cognitive and analytial abilities of the students, \
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the students ability.
Quiz_mCQs:
{quiz}

check from an expert English Writer of the above quiz :


"""


quiz_evaluation_prompt = PromptTemplate(input_variables=["subject", "quiz"], template=template2)

review_chain = LLMChain(llm=llm,prompt=quiz_evaluation_prompt,output_key="review",verbose=True)

generate_evaluate_chain = SequentialChain(chains= [quiz_chain,review_chain], input_variables=["text","number","subject","tone","response_json"],output_variables=["quiz","review"],verbose = True)

 
