from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI

client = OpenAI()

class Summarizer:
    
    @staticmethod
    def summarize_the_pdf(
        file_dir: str,
        max_final_token: int,
        token_threshold: int,
        gpt_model: str,
        temperature: float,
        summarizer_llm_system_role: str,
        final_summarizer_llm_system_role: str,
        character_overlap: int
    ):
        
        docs = []
        docs.extend(PyPDFLoader(file_dir).load())
        print(f"Document length : {len(docs)}")
        max_summarizer_output_token = int(
            max_final_token/len(docs)) - token_threshold
        full_summary = ""
        counter = 1
        print("Generating summary...")
        
        if len(docs) > 1:
            for i in range(len(docs)):
                if i == 0:
                    prompt = docs[i].page_content + \
                        docs[i+1].page_content[:character_overlap]
                        
                elif i < len(docs):
                    prompt = docs[i-1].page_content[-character_overlap:] + \
                        docs[i].page_content + \
                        docs[i+1].page_content[:character_overlap]
                else:  
                    prompt = docs[i-1].page_content[-character_overlap:] + \
                        docs[i].page_content
                summarizer_llm_system_role = summarizer_llm_system_role.format(
                    max_summarizer_output_token)
                
                full_summary += Summarizer.get_llm_response(
                    gpt_model,
                    temperature,
                    summarizer_llm_system_role,
                    prompt = prompt
                )
                
        else:
            full_summary = docs[0].page_content
            
            print(f"Page {counter} was summarized. ", end="")
            counter += 1
            
        final_summary = Summarizer.get_llm_response(
            gpt_model,
            temperature,
            final_summarizer_llm_system_role,
            prompt=full_summary
        )
        return final_summary
    
    @staticmethod
    def get_llm_response(gpt_mode:str,
                         temperature:float,
                         llm_system_role:str,
                         prompt:str):
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", "content": llm_system_role,
                    "role": "user", "content": prompt
                }
            ],
            temperature=temperature
        )
        
        return response.choices[0].message.content