from typing import List
import re
import html
import os
from langchain.schema import Document  

class DataCleaning:
    
    @staticmethod
    def clean_reference(documents: List[Document]) -> str:
        
        markdown_documents = ""
        counter = 1
        
        for doc in documents:
            content = doc.page_content
            metadata_dict = doc.metadata

            content = re.sub(r'\\n', '\n', content)
            content = re.sub(r'\s*<EOS>\s*<pad>\s*', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
            content = html.unescape(content)

            replacements = {
                'â': '-', 'â': '∈', 'Ã': '×',
                'ï¬': 'fi', 'Â·': '·', 'ï¬': 'fl'
            }
            for bad, good in replacements.items():
                content = content.replace(bad, good)

            source = os.path.basename(metadata_dict.get("source", "Unknown file"))
            page = metadata_dict.get("page", "N/A")

            markdown_documents += (
                f"# Retrieved content {counter}:\n"
                f"{content}\n\n"
                f"Source: {source} | Page number: {page}\n\n"
            )
            counter += 1
            
        return markdown_documents
