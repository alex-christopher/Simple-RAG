from openai import OpenAI
import yaml
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from typing import List, Tuple
from utils.load_config import LoadConfig
from utils.data_cleaning import DataCleaning

APPCFG = LoadConfig()

client = OpenAI()

with open("../configs/app_config.yml") as cfg:
    app_config = yaml.load(cfg, Loader=yaml.FullLoader)
    
embedding = OpenAIEmbeddings()

vectordb = Chroma(
    persist_directory=APPCFG.persist_directory,
    embedding_function=embedding
)

while True:
    question = input("\n\n Enter your question or press q to exit \n")
    if question.lower() == "q":
        break
    
    question = "#user new question: \n" + question
    docs = vectordb.similarity_search(question, k=APPCFG.k)
    retrived_docs_page_content: List[Tuple] = [
        str(x.page_content)+"\n\n" for x in docs
    ]
    
    clean_data = DataCleaning.clean_reference(docs)
    print(clean_data)
        
    retrived_docs_str = "#Retrived content: \n\n" + str(retrived_docs_page_content)
    prompt = retrived_docs_str + "\n\n" + question
    
    response = client.chat.completions.create(
        model=APPCFG.llm_engine,
        messages=[
            {
                "role": "system", "content": APPCFG.llm_system_role
            },
            {
                "role": "user", "content": prompt
            }
        ],
        temperature=APPCFG.temperature,
        stream=False
    )
    
    print(response.choices[0].message.content)