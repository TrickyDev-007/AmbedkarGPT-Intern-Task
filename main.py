import os
import sys
import chromadb

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA


def build_vector_db(speech_path: str, db_path: str):
    print("Loading speech file...")
    loader = TextLoader(speech_path, encoding="utf-8")
    docs = loader.load()

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    print(f"Total chunks created: {len(chunks)}")

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Building Chroma vector database...")
    vectordb = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=db_path
    )

    vectordb.persist()
    print("Vector DB created and saved.")

    return vectordb


def load_vector_db(db_path: str):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    return vectordb


def start_qa_system(vectordb):
    print("Loading Mistral model from Ollama...")
    llm = Ollama(model="mistral")

    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    print("\nSystem ready. Ask your questions about the speech.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        result = qa_chain(query)
        answer = result["result"]
        sources = result["source_documents"]

        print("\nAnswer:")
        print(answer)

        print("\nSources:")
        for i, src in enumerate(sources):
            print(f"Chunk {i+1}: {src.page_content[:200]}...")

        print("\n--------------------------------------------\n")


def main():
    speech_path = "speech.txt"
    db_path = "chroma_db"

    if not os.path.exists(speech_path):
        print("ERROR: 'speech.txt' not found in project folder.")
        sys.exit(1)

    # If DB exists, load it. If not, create it.
    if os.path.exists(db_path):
        print("Loading existing vector DB...")
        vectordb = load_vector_db(db_path)
    else:
        vectordb = build_vector_db(speech_path, db_path)

    start_qa_system(vectordb)


if __name__ == "__main__":
    main()
