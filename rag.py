from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA # type: ignore

print("loading PDF...")
loader = PyPDFLoader("pdfs/sample1.pdf")
pages = loader.load()
print(f"Total Number of Pages: {len(pages)}")


print("splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=splitter.split_documents(pages)
print(f"Total number chunks: {len(chunks)}")

print("Creating embeddings...")
embeddings = OpenAIEmbeddings(
    openai_api_key="OPENROUTER_API_KEY",
    openai_api_base="https://openrouter.ai/api/v1",model="qwen/qwen3-embedding-8b")


vectordb=Chroma.from_documents(chunks,embeddings)
print("Stored in Chroma")

llm=ChatOpenAI(
    openai_api_key="OPENROUTER_API_KEY1",
    openai_api_base="https://openrouter.ai/api/v1",model="qwen/qwen3.6-plus:free"
)

qa_chain=RetrievalQA.from_chain_type(
    llm=llm,retriever = vectordb.as_retriever()
)

while True:
    question = input("\nAsk a question (or type 'exit' to quit): ")
    
    if question.lower() == "exit":
        print("Bye!")
        break
    
    answer = qa_chain.invoke(question)
    print(f"\nAnswer: {answer['result']}")