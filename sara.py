# -*- coding: utf-8 -*-

# ---- Dependencies ----
from pypdf import PdfReader
from io import BytesIO
import re
import requests
import logging
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---- Data Loading ----
text_sources = {
    'Garuda': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/GarudaPurana.pdf',
    'Bhagavad Gita': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/Bhagavad-gita_As_It_Is.pdf',
}

pdf_data = {}
for name, url in text_sources.items():
    response = requests.get(url)
    pdf_data[name] = response.content

# ---- Document Extraction ----
documents = []
for name, pdf_bytes in pdf_data.items():
    reader = PdfReader(BytesIO(pdf_bytes))
    for i, page in enumerate(reader.pages):
        documents.append(
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": name, "page": i}
            )
        )

# ---- Cleaning ----
logging.basicConfig(level=logging.INFO)

legal_stopwords = {
    "the","herein", "hereinafter", "therein", "thereof", "thereby", "therewith",
    "whereas", "wherein", "whereof", "thereafter", "thereunder", "hereunder",
    "hereafter", "hereto", "thereto", "hereby", "thereby", "aforementioned",
    "notwithstanding", "pursuant", "agreement", "party", "parties", "contract",
    "terms", "conditions", "plaintiff", "defendant", "court", "judge", "jury",
    "proceedings", "appeal", "case", "order", "decision", "shall", "may", "must",
    "will"
}

def clean_legaltext(text):
    if not text or not isinstance(text, str):
        return ' '
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\\n\\w\\s]', ' ', text)
    words = [
        word for word in text.split() if word not in legal_stopwords or len(word) > 2
    ]
    return ' '.join(words)

# ---- Preprocess ----
def preprocess_documents(documents):
    preprocessed_docs = []
    for doc in documents:
        page_content = doc.page_content if isinstance(doc.page_content, str) else ""
        cleaned_text = clean_legaltext(page_content)
        new_doc = Document(page_content=cleaned_text, metadata=doc.metadata)
        preprocessed_docs.append(new_doc)
    return preprocessed_docs

processed_documents = preprocess_documents(documents)

# ---- Chunking ----
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def chunk_documents(documents):
    chunks = []
    for doc in documents:
        text_chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(text_chunks):
            new_metadata = doc.metadata.copy()
            new_metadata["chunk_id"] = i
            chunks.append(Document(page_content=chunk, metadata=new_metadata))
    return chunks

documents_chunked = chunk_documents(processed_documents)

# ---- Embeddings ----
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ---- Vectorstore ----
vectorstore = FAISS.from_documents(documents_chunked, embedding_function)

# ---- LLM ----
llm_pipeline = pipeline(
    "text-generation",
    model="tiiuae/falcon-7b-instruct",
    max_new_tokens=300,
    temperature=0.5,
    do_sample=True
)

llm = HuggingFacePipeline(pipeline=llm_pipeline)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# ---- FUNCTION FOR DEPLOYMENT ----
def ask(query: str):
    """Returns the model's answer for a given query"""
    result = rag_chain.run(query)
    return result

