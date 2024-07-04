import argparse
import warnings

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

warnings.filterwarnings("ignore", category=DeprecationWarning)


llm = OpenAI()

code_promt = PromptTemplate(
    template="Write a {language} function that will {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=llm, prompt=code_promt, output_key="code")

test_promt = PromptTemplate(
    template="Write a {language} test, for the following code:\n{code}",
    input_variables=["language", "code"],
)

test_chain = LLMChain(llm=llm, prompt=test_promt, output_key="test")

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=["language", "task"],
    output_variables=["test", "code"],
)

result = chain({"language": args.language, "task": args.task})
print(">>>>>")
print(result["code"])
print(">>>>>")
print(result["test"])
