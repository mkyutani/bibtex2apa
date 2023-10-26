import os
import sys
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

load_dotenv()
openai_key = os.environ['OPENAI_API_KEY']

llm = ChatOpenAI(model_name='gpt-3.5-turbo')

template = '''
Convert BibTeX style reference to APA style.

### Output rules
1. If the input reference text includes 'URL' tag, insert the URL at the end of the output in the format "URL: url". Otherwise not insert "URL" text.
2. Do not use line feeds, use white space instead.

### Input

BibTex style reference: {reference}

### Output format

APA Style: 
'''

prompt = PromptTemplate(
    input_variables=['reference'],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt, verbose=False)

print(chain(sys.stdin.read())['text'])
