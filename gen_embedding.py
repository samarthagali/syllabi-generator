import pandas as pd
import faiss
import ollama

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter as rcts
from langchain.schema import Document
import pickle
df=pd.read_csv('Coursera.csv')
df['combined_text']=df[['Course Name', 'Difficulty Level', 'Course Description', 'Skills']].apply(
    lambda row: '\n'.join(row.values.astype(str)), axis=1)
docs=[Document(page_content=text)for text in df['combined_text'].tolist()]

text_splitter=rcts(chunk_size=1000,chunk_overlap=200)
splits=text_splitter.split_documents(docs)
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings=model.encode([doc.page_content for doc in splits])
dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, "faiss_coursera_index.bin")

# Save the document data for retrieval

with open("documents.pkl", "wb") as f:
    pickle.dump(splits, f)
