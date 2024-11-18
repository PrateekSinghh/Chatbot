from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts.lang import tts_langs
import streamlit as st
from gtts import gTTS
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import docx
from langchain_community.vectorstores import FAISS
# import sentence_transformers

# Initialize conversation history

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
    
if 'past' not in st.session_state:
    st.session_state['past'] = []
    
if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""
    
if "conversation" not in st.session_state:
    st.session_state.conversation = None
    
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None
    
if 'audio_files' not in st.session_state:
    st.session_state.audio_files = []


import os 
from dotenv import load_dotenv
load_dotenv()

gemini_api_key =os.getenv('api_key')
if not gemini_api_key:
    raise ValueError("API key not found. Please check your .env file and variable name.")



# Function to format chat history as text for download

def format_chat_for_download(chat_history):
    formatted_text = ""
    for i, message in enumerate(chat_history):
        if message ["role"] == "user":
            formatted_text += f"User: {i+1}: {message['content']}\n"
        elif message["role"] == "bot":
            formatted_text += f"Bot {i+1}: {message['content']}\n"
    return formatted_text

# Function to display conversation and play audio

def display_conversation_and_audio():
    for i, message in enumerate(st.session_state.conversation_history):
        if isinstance(message, dict):
            if message["role"] == "user":
                st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html= True)
            elif message["role"] == "bot":
                st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                response_audio_file = f"response_audio_{(i//2)+1}.mp3"
                st.audio(response_audio_file)
            

# Apply Custom CSS

css = '''
<style>
    /* Avatar image styling */
    .avatar {
        text-align: center;
        margin-top: 2rem;
    }
    
    .avatar img {
        max-height: 250px;
        max-width: 250px;
        object-fit: contain;
        border-radius: 10px;
        border: 2px solid #4a4a4a;
    }

    /* Title styling */
    .title {
        text-align: center;
        font-size: 3rem;
        color: #4a4a4a;
        margin-top: 1rem;
        font-weight: bold;
    }

    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
    }
    .chat-message.user {
        background-color: #e3a9d5;
    }
    .chat-message.bot {
        background-color: #d689cb;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #1c0202;
    }
    body {
        background-color: skyblue !important;
    }
    [data-testid="stSidebar"] {
    }
</style>
'''

# Add the CSS to the Streamlit app
st.markdown(css, unsafe_allow_html=True)

# Display the avatar image at the top
st.markdown('''
    <div class="avatar">
        <img src="https://tile.loc.gov/image-services/iiif/service:ll:llscd:57026883:00010000/full/pct:100/0/default.jpg" 
            style="max-height: 250px; 
            max-width: 250px; 
            object-fit: contain; 
            border-radius: 10px; 
            border: 2px solid #4a4a4a;">
    </div>
''', unsafe_allow_html=True)

# Display the title below the image
st.markdown('<h1 class="title">🎙️ConvoConstitution🤖</h1>', unsafe_allow_html=True)

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" 
            style="max-height: 70px; 
            max-width: 50px; 
            border-radius: 50%; 
            object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1" 
            style="max-height: 80px; 
            max-width: 50px; 
            border-radius: 50%; 
            object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# File upload and processing
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        _, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            st.error(f"Unsupported file type:  {file_extension}. Only PDF and DOCX are supported.")
    return text   

def get_pdf_text(pdf):
    try:
        reader = PdfReader(pdf)
        return "".join([page.extract_text() for page in reader.pages if page.extract_text() ])
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""
    
def get_docx_text(doc_file):
    try:
        doc = docx.Document(doc_file)
        return ' '.join([pare.text for pare in doc.paragraphs if pare.text])
    except Exception as e:
        st.error(f"Error reading DOCX file: {e}")
        return ""
    
def get_text_chunks(text):
        if len(text) < 100:
            st.warning("Document content too short for meaningful Q/A.")
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1500, chunk_overlap=500)
        chunks  = splitter.split_text(text)
        return chunks
    
def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None
    
# Setup conversation chain
def get_conversation_chain(vectorstore, api_key):
    try:
        model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = model,
            retriever=vectorstore.as_retriever(),
        )
    
        return conversation_chain
    except Exception as e:
        st.error(f"Error setting up conversation chain: {e}")
        return None
    
def handle_user_input(user_question):
    response_container = st.container()
    
    try:
        response = st.session_state.conversation.invoke({
            'question': user_question,
            'chat_history': st.session_state.chat_history
        })
    
        st.session_state.chat_history = response['chat_history']
        bot_response = response['answer']
        st.write(f"Bot Response: {response['answer']}")  # Debugging response

        st.markdown(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)

         # Add button to listen to the response
        
        tts = gTTS(bot_response, lang='en')  # You can change the language code if necessary
        temp_audio_dir = "temp_audio"
        os.makedirs(temp_audio_dir, exist_ok=True)
        audio_file = os.path.join(temp_audio_dir,f"response_audio_{len(st.session_state.chat_history) // 2 + 1}.mp3")
            
        # Save the TTS audio to a file in memory
        tts.save(audio_file)
            
        # Play the audio
        st.audio(audio_file)
    except Exception as e:
        st.error(f"Error handling user input: {e}")


    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", messages.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", messages.content), ussafe_allow_html=True)

api_key = gemini_api_key 

# Sidebar for language selection and   Q/A type

language = st.sidebar.selectbox("Select Language", ["Hindi", "English"])
option = "Document Q/A"

if st.sidebar.button("Clear Chat"):
    st.session_state.conversation_history = []
    st.success("Chat history cleared.")
    
if language == "Hindi":
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Please always respond to user queries in Hindi."),
            ("human", "{human_input}"),
        ]
    )
    response_lang = "hi"
else:
    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI assistant. Please always respond to user queries in English."),
            ("human", "{human_input}"),
        ]
    )
    response_lang = "en"
    
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
chain = chat_template | model | StrOutputParser()
            
# Document Q/A Flow

if option == "Document Q/A":
    st.subheader("Please Upload Document")
    uploaded_files = st.file_uploader("Upload PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
            
    if uploaded_files:
        text = get_files_text(uploaded_files)
        if text.strip():
            text_chunks = get_text_chunks(text)
            vector_store = get_vectorstore(text_chunks)
            if vector_store:
                st.session_state.conversation = get_conversation_chain(vector_store, gemini_api_key)
                st.session_state.processComplete = True
                st.success("Document processed successfully!")
            else:
                st.error("Failed to process document for Q/A.")
        else:
            st.warning("No content found in the uploaded files.")
            
    #question = st.text_input("Ask a question about your documents")
    
    question = ""
    if language == "Hindi":
        question = speech_to_text(language="hi",use_container_width=True,just_once=True ,key = "STT_Hindi")
    else:
        question = speech_to_text(language="en",use_container_width=True,just_once=True ,key = "STT_English")

    question = question or st.text_input("Ask a question abour your document")

    if question and st.session_state.conversation:
        handle_user_input(question)
    elif not st.session_state.conversation:
        st.warning("Please upload and process a document first.")
        
    # Download chat history
    if st.download_button("Download Chat History", data=format_chat_for_download(st.session_state.conversation_history).encode("utf-8"), file_name="chat_history.txt", mime="text/plain"):
        st.success("Download started!")
                