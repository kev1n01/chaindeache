import os
from typing import List
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as Pine
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory.buffer import ConversationBufferMemory
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import chainlit as cl
from chainlit.types import AskFileResponse
from io import BytesIO
from chainlit.element import ElementBased


load_dotenv()
pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)

if 'deache-index' not in pc.list_indexes().names():
    pc.create_index(
        name='deache-index',
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index_name = "deache-index"
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
embeddings = OpenAIEmbeddings()

namespaces = set()

welcome_message = """Bienvenido a DeacheChat! 🚀🤖 
Puedes realizar tu consultas cuando quieras
"""


def process_file(file: AskFileResponse):
    if file.type == "text/plain":
        Loader = TextLoader
    elif file.type == "application/pdf":
        Loader = PyPDFLoader

        loader = Loader(file.path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file : AskFileResponse = None):
    if file:
        docs = process_file(file)
        cl.user_session.set("docs", docs)
        docsearch = Pine.from_documents(
            docs, embeddings, index_name=index_name
        )
    else:
        docsearch = Pine.from_existing_index(
            index_name=index_name, embedding=embeddings
        )
    pc.from_documents
    return docsearch


@cl.on_chat_start
async def start():
    if 'deache-index' not in pc.list_indexes().names():
        files = None
        while files is None:
            files = await cl.AskFileMessage(
                content=welcome_message,
                accept=["text/plain", "application/pdf"],
                max_size_mb=20,
                timeout=180,
            ).send()

        file = files[0]

        msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=False, author="You", 
        type="user_message",)
        await msg.send()

        # No async implementation in the Pinecone client, fallback to sync
        docsearch = await cl.make_async(get_docsearch)(file)
    else:
        msg = cl.Message(content=welcome_message,author="Deache", 
        type="assistant_message",)
        await msg.send()
        docsearch = await cl.make_async(get_docsearch)(None)
        print(docsearch)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    if 'deache-index' not in pc.list_indexes().names():
        # Let the user know that the system is ready
        msg.content = f"`{file.name}` processed. You can now ask questions!"
        await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    # source_documents = res["source_documents"]  # type: List[Document]

    # text_elements = []  # type: List[cl.Text]

    # if source_documents:
    #     for source_idx, source_doc in enumerate(source_documents):
    #         source_name = f"source_{source_idx}"
    #         # Create the text element referenced in the message
    #         text_elements.append(
    #             cl.Text(content=source_doc.page_content, name=source_name)
    #         )
    #     source_names = [text_el.name for text_el in text_elements]

        # if source_names:
        #     answer += f"\nSources: {', '.join(source_names)}"
        # else:
        #     answer += "\nNo sources found"

    await cl.Message(content=answer).send()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    if chunk.isStart:
        buffer = BytesIO()
        # This is required for whisper to recognize the file type
        buffer.name = f"input_audio.{chunk.mimeType.split('/')[1]}"
        # Initialize the session for a new audio stream
        cl.user_session.set("audio_buffer", buffer)
        cl.user_session.set("audio_mime_type", chunk.mimeType)

    # Write the chunks to a buffer and transcribe the whole audio at the end
    cl.user_session.get("audio_buffer").write(chunk.data)

@cl.on_audio_end
async def on_audio_end(elements: list[ElementBased]):
    audio_buffer: BytesIO = cl.user_session.get("audio_buffer")
    audio_buffer.seek(0) 
    audio_file = audio_buffer.read()
    audio_mime_type: str = cl.user_session.get("audio_mime_type")
    
    input_audio_el = cl.Audio(
        mime=audio_mime_type, content=audio_file, name=audio_buffer.name
    )
    await cl.Message(
        author="You", 
        type="user_message",
        content="",
        elements=[input_audio_el, *elements]
    ).send()