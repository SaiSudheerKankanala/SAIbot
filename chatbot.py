# chatbot.py
import requests
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from tqdm import tqdm

# ---------------------------
# Sources
# ---------------------------
text_sources = {
    'I.P.C': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/ipc.pdf',
    'Constitution': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/indian%20constitution.pdf',
    'Garuda': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/GarudaPurana.pdf',
    'Bhagavad Gita': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/Bhagavad-gita_As_It_Is.pdf',
    'Quran': 'https://github.com/SaiSudheerKankanala/SAIbot/raw/main/quran-allah.pdf',
    "Bible" : "https://github.com/SaiSudheerKankanala/SAIbot/blob/main/bb.pdf"
}

# ---------------------------
# Download + Load PDFs
# ---------------------------
def load_pdf(url, source_name):
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    for d in docs:
        d.metadata["source"] = source_name
    return docs

print("📚 Loading documents...")
all_docs = {}
for name, url in tqdm(text_sources.items()):
    all_docs[name] = load_pdf(url, name)

# ---------------------------
# Split documents
# ---------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

# ---------------------------
# Build FAISS vector DBs
# ---------------------------
vector_dbs = {}
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("🔎 Creating embeddings...")
for name, docs in tqdm(all_docs.items()):
    chunks = text_splitter.split_documents(docs)
    if chunks:
        vector_dbs[name] = FAISS.from_documents(chunks, embedding=embedder)

# ---------------------------
# Local LLM (FLAN-T5)
# ---------------------------
print("🤖 Loading model...")
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def generate_answer(question, context):
    prompt = f"""
    You are a helpful assistant.
    Answer the question ONLY if it is clearly answered in the given context.
    If the context is unrelated or unclear, respond with exactly: "Not mentioned in this source."

    Question: {question}
    Context: {context}

    Answer:
    """
    return generator(prompt, max_new_tokens=200, clean_up_tokenization_spaces=True)[0]['generated_text']

# ---------------------------
# QA Function
# ---------------------------
def answer_question(question, k=2):
    final_output = []
    for source_name in text_sources.keys():
        if source_name in vector_dbs:
            docs = vector_dbs[source_name].similarity_search(question, k=k)
            if docs:
                context = "\n\n".join([doc.page_content for doc in docs])
                answer = generate_answer(question, context).strip()
                if answer and answer != "Not mentioned in this source.":
                    final_output.append(f"According to {source_name.capitalize()}: {answer}")
    return "\n\n".join(final_output) if final_output else "No relevant answer found in any source."

def chat(question: str):
    return answer_question(question)
