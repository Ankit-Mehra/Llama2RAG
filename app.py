"""
App for Llama Index on Streamlit
"""
# Import streamlit for app dev
import streamlit as st
# Import transformer classes for generaiton
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TextStreamer, pipeline ,StoppingCriteriaList)
# Import torch for datatype attributes
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in stuff to change service context
from llama_index import VectorStoreIndex,ServiceContext,download_loader,set_global_service_context
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from torch import cuda
from environs import Env
from stop_token import StopOnTokens
import time


# get hugging face token from .env file
env = Env()
env.read_env(path='.env')
HG_TOKEN = env.str("hugging_face_token")

MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'
DEVICE = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

@st.cache_resource
def get_tokenizer_model():
    """
    get tokenizer and model
    """
    # Create tokenizer
    pre_trained_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID,
                                              cache_dir='./model/',
                                              token=HG_TOKEN)

    # Create model
    per_trained_model = AutoModelForCausalLM.from_pretrained(MODEL_ID,
                                                 cache_dir='./model/',
                                                 token=HG_TOKEN,
                                                 torch_dtype=torch.float16,
                                                 rope_scaling={"type": "dynamic", "factor": 2},
                                                 load_in_8bit=True,
                                                 device_map ="auto") 

    return pre_trained_tokenizer, per_trained_model
tokenizer, model = get_tokenizer_model()

# Create a system prompt
SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as 
helpfully as possible, while being safe. Your answers should not include
any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain 
why instead of answering something not correct. If you don't know the answer 
to a question, please don't share false information.

Your goal is to provide answers relating to the financial performance of 
the company.<</SYS>>
"""
# Throw together the query wrapper
query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

# Create a HF LLM using the llama index wrapper
# llm = HuggingFaceLLM(context_window=4096,
#                     max_new_tokens=256,
#                     SYSTEM_PROMPT=SYSTEM_PROMPT,
#                     query_wrapper_prompt=query_wrapper_prompt,
#                     model=model,
#                     tokenizer=tokenizer)

# Setup the text streamer
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

# Setup stop list
stop_list = ['\nHuman:', '\n```\n']

# Setup stopping criteria
stopping_criteria = StoppingCriteriaList([StopOnTokens(tokenizer = tokenizer,
                                 stop_list = stop_list,
                                 device = DEVICE)])

# Create a text generation pipeline
generate_text = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    streamer=streamer,
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.7,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens= 1024,# max number of tokens to generate in the output
    repetition_penalty=1.1,  # without this output begins repeating
    # SYSTEM_PROMPT=SYSTEM_PROMPT,
    # query_wrapper_prompt=query_wrapper_prompt,
    # content_window=4096
)

# Create a new LLM instance using the pipeline
llm = HuggingFacePipeline(pipeline=generate_text)

# Create and dl embeddings instance
embeddings=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Create new service context instance
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embeddings
)
# And set the service context
set_global_service_context(service_context)

# Download PDF Loader
PyMuPDFReader = download_loader("PyMuPDFReader")

# Create PDF Loader
loader = PyMuPDFReader()
# Load documents
documents = loader.load(file_path='./data/handsOn.pdf', metadata=True)

# Create an index - we'll be able to query this in a sec
index = VectorStoreIndex.from_documents(documents)

# Setup index query engine using LLM
query_engine = index.as_query_engine()

# Create centered main title
st.title('🦙 Llama Your ML Instructor')

# Create a text input box for the user
prompt = st.chat_input("Question me about AI and I'll answer!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        assistant_response = query_engine.query(prompt)
        # Simulate stream of response with milliseconds delay
        for chunk in assistant_response.response.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        with st.expander('Response Object'):
            st.write(assistant_response)
        # Display source text
        with st.expander('Source Text'):
            st.write(assistant_response.get_formatted_sources())
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
