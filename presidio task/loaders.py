from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings


def loadfromWeb():
        loader = WebBaseLoader('https://www.cpedv.org/who-we-are')
        webcontent = loader.load()
        webdoc = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100).split_documents(webcontent)
        webVectorDb = FAISS.from_documents(webdoc, OllamaEmbeddings(model='nomic-embed-text'))
        webRetriver = webVectorDb.as_retriever()
        return webRetriver

def loadfrompdf(path):
        pdfloader = PyPDFLoader(path)
        pdfcontent = pdfloader.load()
        pdfdoc = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100).split_documents(pdfcontent)
        pdfvectordb = FAISS.from_documents(pdfdoc, OllamaEmbeddings(model='nomic-embed-text'))
        pdfRetriver = pdfvectordb.as_retriever()
        return pdfRetriver

