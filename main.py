

"""
PDF-based QA system using vector search and OpenAI's GPT. 
Processes Japanese PDFs, indexes them with FAISS, and answers queries via a Gradio UI.
"""

import faiss
import gradio as gr
import os
from transformers import AutoTokenizer
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import fitz
import json

# === Setup ===
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# === Constants ===
FOLDER_PATH = "documents/"
EMBEDDINGS_PATH = "embeddings.npy"
DOCS_PATH = "documents.json"
MODEL_NAME = "BAAI/bge-m3"  # Using BAAI's BGE model for better multilingual support

# === Globals ===
index = None  
id_to_metadata = {}

# === Models ===
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# === Utility Functions ===
def chunk_by_bge_tokens(text, chunk_size=300, overlap=50):
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(input_ids):
        end = start + chunk_size
        chunk_ids = input_ids[start:end]
        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append(chunk_text)
        start += chunk_size - overlap
    return chunks

def create_index(vectors: np.ndarray,):
    global index 
    # Create FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    data = np.vstack(vectors).astype('float32')
    # Add to index with IDs
    index.add(data)
    size_in_bytes = faiss.serialize_index(index).nbytes
    size_in_mb = size_in_bytes / (1024 * 1024)
    print(f"FAISS index size in memory: {size_in_mb:.2f} MB")
    return index

def chunk_data(document):
    # Chunk document
    chunks = chunk_by_bge_tokens(document)
    # Generate embedding 
    embedding_list = []
    print("No of chunks: ", len(chunks))
    for chunk in chunks:
        embedding = model.encode(chunk)  # This returns a 2D numpy array
        embedding = np.array(embedding, dtype='float32')
        embedding_list.append(embedding)
    return chunks, embedding_list

def preprocess_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    result = "\n".join([page.get_text() for page in doc]) 
    return result

# Load and process PDF files, or load processed JSON data
def load_data():
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r") as f:
            docs = json.load(f)
        print(type(docs))
        result_list = docs
    else:
        files = [f"{FOLDER_PATH}{f}" for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))]
        result_list = []
        for doc in files:
            result = preprocess_pdf(doc)
            result_list.append(result)
        with open(DOCS_PATH, "w") as f:
            json.dump(result_list, f, ensure_ascii=False, indent=2)
              
    return result_list

# Call OpenAI API, send relevant doc, get it summarised 
def create_prompt(context_text, user_question):

    prompt_template = f"""You are a helpful assistant. You will be given two inputs:

        1. A block of context text extracted from internal documents
        2. A user question

        Your task is to:
        - Summarize the context in clear and concise Japanese
        - Then, using that summary, answer the user’s question as accurately and helpfully as possible

        If the context is not relevant to the question, say so clearly.

        Respond in the following format:

        【要約】
        <summary in Japanese>

        【回答】
        <answer to the question in Japanese>
        
         ---  
        【Context】  
        {context_text} 

        【User Question】  
        {user_question} """
    
    return prompt_template

def dense_retrieval(user_question):
    global index
    embedded_question = model.encode(user_question)
    query_vector = np.array(embedded_question).astype('float32').reshape(1, -1)
    _, indices = index.search(query_vector, 5)
    retrieved_indices = indices[0]  # this is a list of 5 indices
    retrieved_chunks = [id_to_metadata[i]["chunk"] for i in retrieved_indices]
    print("Retrieved docs:", retrieved_chunks)
    return retrieved_chunks

def call_llm(user_question):
    context_texts = dense_retrieval(user_question)
    request = create_prompt("\n\n".join(context_texts), user_question)
    response = client.responses.create(
        model="gpt-4.1",
        input=request
    )
    print(response.output_text)
    response = f"Backend response to: {user_question}.\n {response.output_text}"
    return response

def generate_embeddings_and_create_index(documents):
    """
    Builds a single (N, d) matrix, assigns stable int64 IDs, and creates a FAISS index.
    Keeps only lightweight metadata per ID.
    """
    global id_to_metadata, index

    all_vecs = []
    all_ids = []
    next_id = 0

    for doc_id, doc in enumerate(documents):
        # chunks: list[str], vecs: list[np.ndarray (d,)]
        chunks, vecs = chunk_data(doc)

        for chunk_idx, (chunk, v) in enumerate(zip(chunks, vecs)):
            # Ensure 1D float32
            v = np.asarray(v, dtype="float32").reshape(-1)

            # Assign ID, collect metadata 
            id_to_metadata[next_id] = {
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "chunk": chunk,           # keep the text; omit if you’ll load from disk later
            }
            all_ids.append(next_id)
            all_vecs.append(v)
            next_id += 1

    # Stack to (N, d)
    X = np.vstack(all_vecs).astype("float32")
    ids = np.asarray(all_ids, dtype="int64")
    create_index(X)

# === Gradio UI ===
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            user_question = gr.Textbox(label="Enter your query（例：分野横断的な対応が求められる課題の事例を一つ教えてください）")
            submit_button = gr.Button("Submit")
        with gr.Column():
            output = gr.Textbox(label="Generated Output", lines=5)
    submit_button.click(fn=call_llm, inputs=user_question, outputs=output)

if __name__ == "__main__":
    documents = load_data()
    generate_embeddings_and_create_index(documents)
    demo.launch(server_name="0.0.0.0", server_port=7860)
