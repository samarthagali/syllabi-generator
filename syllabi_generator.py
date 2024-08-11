import faiss
import pickle
from sentence_transformers import SentenceTransformer
import ollama
import gradio as gr

# Load the FAISS index
index = faiss.read_index("faiss_coursera_index.bin")

# Load the document data
with open("documents.pkl", "rb") as f:
    splits = pickle.load(f)

# Load the embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Define the function to call the Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}. Please ensure that you only provide the syllabus with no other messages."
    response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the retriever function
def retriever(question):
    question_embedding = model.encode([question])
    _, indices = index.search(question_embedding, 5)  # Retrieve 5 closest documents
    return [splits[i].page_content for i in indices[0]]

# Define the RAG setup
def rag_chain(question):
    retrieved_docs = retriever(question)
    formatted_context = "\n\n".join(retrieved_docs)
    return ollama_llm(question, formatted_context)

# Define the Gradio interface
def get_important_facts(question):
    question="Design a 5 module syllabus for the following course:"+question
    return rag_chain(question)

# Create a Gradio app interface
iface = gr.Interface(
    fn=get_important_facts,
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."),
    outputs="text",
    title="RAG with Llama3 on Coursera Dataset",
    description="Ask questions about the provided Coursera course data",
)

# Launch the Gradio app
iface.launch()
