import argparse
import warnings
from email.policy import default

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

warnings.filterwarnings("ignore", category=DeprecationWarning)

llm = OpenAI()
code_promt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"],
)

code_chain = LLMChain(llm=llm, prompt=code_promt)

result = code_chain({"language": args.language, "task": args.task})
print(result)
