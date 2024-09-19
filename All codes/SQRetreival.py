# Loading the documents from langchain resources folder

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(path_pdf):
    get_text = PyPDFLoader(path_pdf)

    get_pages = get_text.load()

    final_text = []

    shredder = RecursiveCharacterTextSplitter(chunk_size=350,
                                              chunk_overlap=20,
                                              length_function=len)

    final_shred = shredder.split_documents(get_pages)

    return final_shred

#Just to test the function
agent_pg = load_pdf("/content/Agents.pdf")

agent_pg[0]

import glob
file_list = glob.glob("/content/*.pdf")
all_docs = []
for file in file_list:
  temp_docs = load_pdf(file)
  all_docs.extend(temp_docs)

len(all_docs)

import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS']="/content/generativeaitrial-trialLC.json"

from langchain.schema import Document
from langchain.embeddings import VertexAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = VertexAIEmbeddings()

vectordb = ''

persist_directory = 'lc_documentdb'
vectordb = Chroma.from_documents(documents=all_docs,
                  persist_directory=persist_directory,
                  embedding=embeddings)

vectordb.persist()

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embeddings)

#some commented statements here.

db_retriever = vectordb.as_retriever()
db_retriever.get_relevant_documents("langchain concepts")

#!zip -r lc_documentdb.zip /content/lc_documentdb

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info=[
    AttributeInfo(
        name="source",
        description="Filename and location of the source file",
        type="string",
    ),
    AttributeInfo(
        name="page",
        description="Page number on which the document is found",
        type="integer",
    )
]
document_content_description = "Text documents from Langchain help and concept documentation"

from langchain.llms import VertexAI

llm = VertexAI(temperature=0)

retriever = SelfQueryRetriever.from_llm(llm,
                                        vectordb,
                                        document_content_description,
                                        metadata_field_info,
                                        verbose=True,
                                        )

retriever.get_relevant_documents("What are some concepts of Agents")

retriever = SelfQueryRetriever.from_llm(llm,
                                        vectordb,
                                        document_content_description,
                                        metadata_field_info,
                                        verbose=True,
                                        enable_limit=True)

retriever.get_relevant_documents("Explain 3 concepts of Chains")

retriever.get_relevant_documents("Give 2 example of autonomous agent")

from langchain.chains import RetrievalQAWithSourcesChain

chain = RetrievalQAWithSourcesChain.from_chain_type(llm,
                                                    chain_type="stuff",
                                                    retriever=retriever)

chain({"question":"Give 2 types of agents"})