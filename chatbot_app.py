from langchain.chains import LLMChain
from langchain.llms import Bedrock
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate
import streamlit as st
from PIL import Image
import boto3
import base64
import json
from botocore.exceptions import ClientError
import logging
import docx2txt
from pdfminer.high_level import extract_text
from io import BytesIO
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

bedrock = boto3.client(
    service_name='bedrock',
    region_name='us-east-1'
)

st.title("🤖Multi-Modal Chatbot")
st.markdown("Image and File Support Included，powered by Claude3 sonnet on Bedrock")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]


with (st.sidebar):
    uploaded_file = st.file_uploader("上传Word或者PDF文件，或者JPG/PNG图片", accept_multiple_files=False,type=['jpg', 'jpeg', 'png','docx', 'pdf'])
    if uploaded_file:
        file_type = uploaded_file.type
        if file_type == 'application/pdf':
            text = extract_text(uploaded_file)
        elif file_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = docx2txt.process(uploaded_file)
        elif file_type in ['image/jpg', 'image/jpeg', 'image/png']:
            image = Image.open(uploaded_file)
            image = image.convert('RGB')
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()


        else:
            st.error("仅支持上传Word/PDF文件，以及JPG/PNG图片")

    st.divider()

    if st.button("Clear Chat History"):
        st.session_state.messages.clear()
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi there. Can I help you?"}]

def invoke_bedrock_model(bedrock_runtime, model_id, messages, max_tokens):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
             "messages": messages
        }
    )

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body

def invoke_with_system_pe(bedrock_runtime, model_id, max_tokens,system, messages):
    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
             "messages": messages
        }
    )

    response = bedrock_runtime.invoke_model(body=body, modelId=model_id)
    response_body = json.loads(response.get('body').read())

    return response_body

def text_file_prompts(context,history,question):
    try:

        bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name='us-east-1')

        model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        max_tokens = 50000
        prompt_template = f"""Human: This is a friendly conversation between a human and an AI. 
        The AI is talkative and provides specific details from its context.You answer with what language the human use or what human request.
        If the AI does not know the answer to a question, it truthfully says it does not know.
        Assistant: OK, got it, I'll be a talkative truthful AI assistant.
        Human: Here are a few documents in <documents> tags:
        <documents>
        {context}
        </documents>
        Current conversation:
        <conversation_history>
        {history}
        </conversation_history>
        Based on the above documents and history conversation, provide a detailed answer for, {question} 
        Answer "don't know" if not present in the document. Be more rigorous.
        Assistant:"""

        message = {"role": "user","content": [{"type": "text", "text": prompt_template}]}
        messages = [message]
        response = invoke_bedrock_model(bedrock_runtime, model_id, messages, max_tokens)
        # print(json.dumps(response, indent=4))
        return response['content'][0]['text']

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))

def image_file_prompts(content_image,file_type,history,question):
    try:

        bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name='us-east-1')

        model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        max_tokens = 50000
        prompt_template = f"""Human: This is a friendly conversation between a human and an AI. 
        You are an AI chatbot who is talkative, you should respond by identifying the language used by the human or clarifying what the human is asking for.
        If you does not know the answer to a question, it truthfully says it does not know.
        Assistant: OK, got it, I'll be a talkative truthful AI assistant.
        Current conversation:
        <conversation_history>
        {history}
        </conversation_history>
        Based on the image and history conversation, provide a detailed answer for, {question} 
        Answer "don't know" if not present in the document. Be more rigorous.
        Assistant:"""

        message = {"role": "user",
                   "content": [
                       {"type": "image", "source": {"type": "base64",
                                                    "media_type": file_type,
                                                    "data": content_image}},
                       {"type": "text", "text": prompt_template}
                ]}
        messages = [message]
        system = "Respond only in chinese"
        response = invoke_with_system_pe(bedrock_runtime, model_id, max_tokens,system, messages)

        return response['content'][0]['text']

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))
def text_prompts(history,question):
    try:

        bedrock_runtime = boto3.client(service_name='bedrock-runtime',region_name='us-east-1')

        model_id = 'anthropic.claude-3-sonnet-20240229-v1:0'
        max_tokens = 50000
        prompt_template = f"""Human: This is a friendly conversation between a human and an AI. 
        The AI is talkative and answer always in Chinese.If the AI does not know the answer to a question, it truthfully says it does not know.
        Assistant: OK, got it, I'll be a talkative truthful AI assistant.
        Current conversation:
        <conversation_history>
        {history}
        </conversation_history>
        Combine with history conversation,provide answer for, {question} 
        Assistant:"""

        message = {"role": "user","content": [{"type": "text", "text": prompt_template}]}
        messages = [message]
        response = invoke_bedrock_model(bedrock_runtime, model_id, messages, max_tokens)
        # print(json.dumps(response, indent=4))
        return response['content'][0]['text']

    except ClientError as err:
        message = err.response["Error"]["Message"]
        logger.error("A client error occurred: %s", message)
        print("A client error occured: " +
              format(message))

def get_last_5_conversations(msgs):
    all_messages = msgs.messages
    if len(all_messages) <= 5:
        #print(all_messages)
        return all_messages


        # 否则，返回最近五条消息
    last_5_messages = all_messages[-5:]
    return last_5_messages

if uploaded_file:
    if file_type in ['image/jpg', 'image/jpeg', 'image/png']:

        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])


        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            msgs = StreamlitChatMessageHistory(key="messages")
            memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)

            last_5_conversations = get_last_5_conversations(msgs)
            response = image_file_prompts(content_image=image_base64,file_type=file_type,history=last_5_conversations,question=prompt)
            msg = response
            st.session_state.messages.append({"role": "assistant", "content": msg})

            st.image(image, width=300)
            st.chat_message("assistant").write(msg)
    else:

        if "messages" not in st.session_state:
            st.session_state.messages = []
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])


        if prompt := st.chat_input():
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            msgs = StreamlitChatMessageHistory(key="messages")
            memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
            last_5_conversations = get_last_5_conversations(msgs)
            ans = text_file_prompts(context=text,history=last_5_conversations,question=prompt)
            msg = ans
            st.session_state.messages.append({"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)


else:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        msgs = StreamlitChatMessageHistory(key="messages")
        memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)
        last_5_conversations = get_last_5_conversations(msgs)
        ans = text_prompts(history=last_5_conversations,question=prompt)
        msg = ans
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)
