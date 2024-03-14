
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.llms import HuggingFaceHub


from langchain.retrievers import BM25Retriever, EnsembleRetriever

from io import BytesIO
import fitz  # PyMuPDF

import os
def display_res(query,text):
    docs = [Document(page_content=text)]
    # create chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                            chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    
    # Get Embedding Model from HF via API
    HF_TOKEN = "your-token"

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)

    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": 3})

    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k =  3

    ensemble_retriever = EnsembleRetriever(retrievers=[vectorstore_retreiver,
                                                    keyword_retriever],
                                        weights=[0.5, 0.5])

    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        model_kwargs={"temperature": 0.3,"max_new_tokens":1024},
        huggingfacehub_api_token=HF_TOKEN,
    )

    template = """

    <|system|>>
    You are a helpful AI Assistant that follows instructions extremely well.
    Use the following context to answer user question.

    Think step by step before answering the question. You will get a $100 tip if you provide correct answer.

    CONTEXT: {context}
    </s>
    <|user|>
    {query}
    </s>
    <|assistant|>
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = (
        {"context": ensemble_retriever, "query": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    res = chain.invoke(query)
    print(res)
    return res
def extract_text_from_pdf(pdf_content):
    text= ''
    with BytesIO(pdf_content) as file_buffer:
        pdf_document = fitz.open(stream=file_buffer.read(), filetype="pdf")
        print(pdf_document.page_count)
        for page_num in range(pdf_document.page_count):
            page = pdf_document[page_num]
            text += page.get_text()
    return text