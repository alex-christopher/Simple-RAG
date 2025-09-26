from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

import os

class PrepareVectorDB:
    
    def __init__(
        self,
        data_directory: str,
        persist_directory: str,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int
    ):
        
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.data_directory = data_directory
        self.persist_directory = persist_directory
        self.embedding = OpenAIEmbeddings(model=embedding_model)
        
    def __load_all_documents(self):
        
        doc_counter = 0
        if isinstance(self.data_directory, list):
            print("Loading uploaded docs...")
            docs = []
            
            for docs_dir in self.data_directory:
                docs.extend(PyPDFLoader(docs_dir).load())
                doc_counter += 1
                
            print("Number of loaded docs : ", doc_counter)
            print("Number of loaded pages : ", len(docs), "\n\n")
            
        else:
            print("Loading docs manually...")
            document_list = os.listdir(self.data_directory)
            docs = []
            
            for doc_name in document_list:
                docs.extend(PyPDFLoader(os.path.join(
                    self.data_directory, doc_name
                )).load())
                doc_counter += 1
            
            print("Number of loaded docs : ", doc_counter)
            print("Number of loaded pages : ", len(docs), "\n\n")
        
        return docs
    
    def __chunk_documents(self, docs):
        
        print("Chunking documents...")
        chunked_docs = self.text_splitter.split_documents(docs)
        print("No of chunks : ", len(chunked_docs), "\n\n")
        
        return chunked_docs
    
    def prepare_and_save_vectordb(self):
        
        docs = self.__load_all_documents()
        chunked_docs = self.__chunk_documents(docs)
        
        print("Preparing vector database...")
        vectordb = Chroma.from_documents(
            documents=chunked_docs,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        
        print("VectorDB created and saved")
        results = vectordb.get()
        num_vectors = len(results["ids"])
        print("Number of vectors in vector db : ", num_vectors, "\n\n")

        return vectordb