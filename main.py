import streamlit as st
import openai
import base64, re
from io import BytesIO
from audio_recorder_streamlit import audio_recorder
from gtts import gTTS
from PIL import Image, UnidentifiedImageError
from langchain.schema import SystemMessage
import initialize , chat_file, chat_image
def read_audio(audio_bytes):
    """
    This function reads audio bytes and returns the corresponding text.
    """
    try:
        audio_data = BytesIO(audio_bytes)
        audio_data.name = "recorded_audio.wav"  # dummy name

        transcript = st.session_state.openai.audio.transcriptions.create(
            model="whisper-1", file=audio_data
        )
        text = transcript.text
    except Exception as e:
        text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return text


def perform_tts(text):
    """
    This function takes text as input, performs text-to-speech (TTS),
    and returns an audio_response.
    """

    try:
        with st.spinner("TTS in progress..."):
            audio_response = st.session_state.openai.audio.speech.create(
                model="tts-1",
                voice=st.session_state.tts_voice,
                input=text,
            )
    except Exception as e:
        audio_response = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return audio_response


def perform_tts2(text):
    try:
        with st.spinner("TTS in progress..."):
            tts = gTTS(text=text, lang='en', tld='com', slow=False)
            audio_response = BytesIO()      # convert to file-like object
            tts.write_to_fp(audio_response)
    except Exception as e:
        audio_response = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return audio_response


def play_audio(audio_response):
    """
    This function takes an audio response (a bytes-like object)
    from TTS as input, and plays the audio.
    """

    if st.session_state.tts_model == "OpenAI":
        audio_data = audio_response.read()

        # Encode audio data to base64
        b64 = base64.b64encode(audio_data).decode("utf-8")

        # Create a markdown string to embed the audio player with the base64 source
        md = f"""
         <audio controls autoplay style="width: 100%;">
         <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
         Your browser does not support the audio element.
            </audio>
         """
         # Use Streamlit to render the audio player
        st.markdown(md, unsafe_allow_html=True)
    elif st.session_state.tts == "gTTS":
        st.audio(audio_response)
            



def is_url(text):
    """
    This function determines whether text is a URL or not.
    """

    regex = r"(http|https)://([\w_-]+(?:\.[\w_-]+)+)(:\S*)?"
    p = re.compile(regex)
    match = p.match(text)
    if match:
        return True
    else:
        return False


def reset_conversation():
    st.session_state.messages = [
        SystemMessage(content=st.session_state.ai_role[0])
    ]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]
    st.session_state.prompt_exists = False
    st.session_state.human_enq = []
    st.session_state.ai_resp = []
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.audio_response = None
    st.session_state.audio_response_gtts = None
    st.session_state.vector_store = None
    st.session_state.sources = None
    st.session_state.memory = None


def switch_between_apps():
    st.session_state.temperature[1] = st.session_state.temperature[0]
    st.session_state.image_source[1] = st.session_state.image_source[0]
    st.session_state.ai_role[1] = st.session_state.ai_role[0]


def enable_user_input():
    st.session_state.prompt_exists = True


def reset_qna_image():
    st.session_state.uploaded_image = None
    st.session_state.qna = {"question": "", "answer": ""}


def create_text(model):
    """
    This function generates text based on user input
    by calling chat_complete().

    model is set to "gpt-3.5-turbo" or "gpt-4".
    """

    # initial system prompts
    general_role = "You are a helpful assistant."
    english_teacher = "You are an English teacher who analyzes texts and corrects any grammatical issues if necessary."
    translator = "You are a translator who translates English into Korean and Korean into English."
    coding_adviser = "You are an expert in coding who provides advice on good coding styles."
    doc_analyzer = "You are an assistant analyzing the document uploaded."
    roles = (general_role, english_teacher, translator, coding_adviser, doc_analyzer)

    with st.sidebar:
        # Text to Speech selection
        st.write("")
        st.write("**Text to Speech**")
        st.session_state.tts = st.radio(
            label="$\\hspace{0.08em}\\texttt{TTS}$",
            options=("Enabled", "Disabled", "Auto"),
            horizontal=True,
            index=1,
            label_visibility="collapsed",
        )
        # TTS model selection
        st.write("")
        st.write("**TTS model**")
        st.session_state.tts_model = st.radio(
            label="$\\hspace{0.08em}\\texttt{MODEL}$",
            options=("OpenAI", "gTTS"),
            horizontal=True,
            index=1,
            label_visibility="collapsed",
        )
        # TTS voice selection
        if st.session_state.tts_model == "OpenAI":
            st.write("")
            st.write("**TTS Voice**")
            st.session_state.tts_voice = st.radio(
                label="$\\hspace{0.08em}\\texttt{MODEL}$",
                options=("alloy", "echo","fable","onyx","nova","shimmer"),
                horizontal=True,
                index=1,
                label_visibility="collapsed",
        )
        # Temperature selection
        st.write("")
        st.write("**Temperature**")
        st.session_state.temperature[0] = st.slider(
            label="$\\hspace{0.08em}\\texttt{Temperature}\,$ (higher $\Rightarrow$ more random)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature[1],
            step=0.1,
            format="%.1f",
            label_visibility="collapsed",
        )
        st.write("(Default=0.7, Higher $\Rightarrow$ More random)")

    st.write("")
    st.write("##### Message to AI")
    st.session_state.ai_role[0] = st.selectbox(
        label="AI's role",
        options=roles,
        index=roles.index(st.session_state.ai_role[1]),
        # on_change=reset_conversation,
        label_visibility="collapsed",
    )

    if st.session_state.ai_role[0] != st.session_state.ai_role[1]:
        reset_conversation()

    if st.session_state.ai_role[0] == doc_analyzer:
        st.write("")
        left, right = st.columns([4, 7])
        left.write("##### Document to ask about")
        right.write("If you want a consistent answer, set the Temperature param to 0.")
        uploaded_file = st.file_uploader(
            label="Upload an article",
            type=["txt", "pdf", "docx", "pptx", "csv", "html"],
            accept_multiple_files=False,
            on_change=reset_conversation,
            label_visibility="collapsed",
        )
        if st.session_state.vector_store is None:
            # Create the vector store.
            st.session_state.vector_store = chat_file.get_vector_store(uploaded_file)

            if st.session_state.vector_store is not None:
                st.write(f"Vector store for :blue[[{uploaded_file.name}]] is ready!")

    st.write("")
    left, right = st.columns([4, 7])
    left.write("##### Conversation with AI")
    right.write("Click on the mic icon and speak, or type text below.")

    # Print conversations
    for human, ai in zip(st.session_state.human_enq, st.session_state.ai_resp):
        with st.chat_message("human"):
            st.write(human)
        with st.chat_message("ai"):
            st.write(ai)

    if st.session_state.ai_role[0] == doc_analyzer and st.session_state.sources is not None:
        with st.expander("Sources"):
            c1, c2, _ = st.columns(3)
            c1.write("Uploaded document:")
            columns = c2.columns(len(st.session_state.sources))
            for index, column in enumerate(columns):
                column.markdown(
                    f"{index + 1}\)",
                    help=st.session_state.sources[index].page_content
                )

    # Play TTS
    if st.session_state.audio_response is not None: # tts_model == "OpenAI"
        play_audio(st.session_state.audio_response)
        st.session_state.audio_response = None

    elif st.session_state.audio_response_gtts is not None: # tts_model == "gTTS"
        st.audio(st.session_state.audio_response_gtts)
        # audio_tag = f'<audio autoplay="true" src="data:audio/wav;base64,{st.session_state.audio_response_gtts}">'
        # st.markdown(audio_tag, unsafe_allow_html=True)
        st.session_state.audio_response_gtts = None

    # Reset the conversation
    st.button(label="Reset the conversation", on_click=reset_conversation)

    # Use your keyboard
    user_input = st.chat_input(
        placeholder="Enter your query",
        on_submit=enable_user_input,
        disabled=not uploaded_file if st.session_state.ai_role[0] == doc_analyzer else False
    )

    # Use your microphone
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )

    if audio_bytes != st.session_state.audio_bytes:
        user_prompt = read_audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes
        if user_prompt is not None:
            st.session_state.prompt_exists = True
            st.session_state.mic_used = True
    elif user_input and st.session_state.prompt_exists:
        user_prompt = user_input.strip()

    if st.session_state.prompt_exists:
        with st.chat_message("human"):
            st.write(user_prompt)

        with st.chat_message("ai"):
            if st.session_state.ai_role[0] == doc_analyzer:
                generated_text, st.session_state.sources = chat_file.document_qna(
                    user_prompt,
                    vector_store=st.session_state.vector_store,
                    model=model
                )
            else:  # General chatting
                generated_text = chat_file.chat_complete(
                    user_prompt,
                    temperature=st.session_state.temperature[0],
                    model=model
                )

        if generated_text is not None:
            # TTS under two conditions
            cond1 = st.session_state.tts == "Enabled"
            cond2 = st.session_state.tts == "Auto" and st.session_state.mic_used

            if cond1 or cond2:
                if st.session_state.tts_model == "OpenAI":
                    st.session_state.audio_response = perform_tts(generated_text)
                elif st.session_state.tts_model == "gTTS":
                    # st.session_state.audio_response = perform_tts2(generated_text)
                    # st.audio(perform_tts2(generated_text))
                    st.session_state.audio_response_gtts = perform_tts2(generated_text)
                    # audio_response = perform_tts2(generated_text)
                    # st.audio(audio_response)
                    
                              
            st.session_state.mic_used = False
            st.session_state.human_enq.append(user_prompt)
            st.session_state.ai_resp.append(generated_text)

        st.session_state.prompt_exists = False

        if generated_text is not None:
           st.rerun()
            

def create_text_with_image(model):
    """
    This function responds to the user's query about the image
    from a URL or uploaded image.
    """

    with st.sidebar:
        sources = ("From URL", "Uploaded")
        st.write("")
        st.write("**Image selection**")
        st.session_state.image_source[0] = st.radio(
            label="Image selection",
            options=sources,
            index=sources.index(st.session_state.image_source[1]),
            label_visibility="collapsed",
        )

    st.write("")
    st.write("##### Image to ask about")
    st.write("")

    if st.session_state.image_source[0] == "From URL":
        # Enter a URL
        st.write("###### :blue[Enter the URL of your image]")

        image_url = st.text_input(
            label="URL of the image", label_visibility="collapsed",
            on_change=reset_qna_image
        )
        if image_url:
            if is_url(image_url):
                st.session_state.uploaded_image = image_url
            else:
                st.error("Enter a proper URL", icon="🚨")
    else:
        # Upload an image file
        st.write("###### :blue[Upload your image]")

        image_file = st.file_uploader(
            label="High resolution images will be resized.",
            type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=False,
            label_visibility="collapsed",
            on_change=reset_qna_image,
        )
        if image_file is not None:
            # Process the uploaded image file
            try:
                image = Image.open(image_file)
                st.session_state.uploaded_image = chat_image.shorten_image(image, 1024)
            except UnidentifiedImageError as e:
                st.error(f"An error occurred: {e}", icon="🚨")

    # Capture the user's query and provide a response if the image is ready
    if st.session_state.uploaded_image:
        st.image(image=st.session_state.uploaded_image, use_column_width=True)

        # Print query & answer
        if st.session_state.qna["question"] and st.session_state.qna["answer"]:
            with st.chat_message("human"):
                st.write(st.session_state.qna["question"])
            with st.chat_message("ai"):
                st.write(st.session_state.qna["answer"])

        # Use your microphone
        audio_bytes = audio_recorder(
            pause_threshold=3.0, text="Speak", icon_size="2x",
            recording_color="#e87070", neutral_color="#6aa36f"        
        )
        if audio_bytes != st.session_state.audio_bytes:
            st.session_state.qna["question"] = read_audio(audio_bytes)
            st.session_state.audio_bytes = audio_bytes
            if st.session_state.qna["question"] is not None:
                st.session_state.prompt_exists = True

        # Use your keyboard
        query = st.chat_input(
            placeholder="Enter your query",
        )
        if query:
            st.session_state.qna["question"] = query
            st.session_state.prompt_exists = True

        if st.session_state.prompt_exists:
            if st.session_state.image_source[0] == "From URL":
                generated_text = chat_image.openai_query_image_url(
                    image_url=st.session_state.uploaded_image,
                    query=st.session_state.qna["question"],
                    model=model
                )
            else:
                generated_text = chat_image.openai_query_uploaded_image(
                    image_b64=chat_image.image_to_base64(st.session_state.uploaded_image),
                    query=st.session_state.qna["question"],
                    model=model
                )

            st.session_state.prompt_exists = False
            if generated_text is not None:
                st.session_state.qna["answer"] = generated_text
                st.rerun()


def create_image(model):
    """
    This function generates image based on user description
    by calling openai_create_image().
    """

    # Set the image size
    with st.sidebar:
        st.write("")
        st.write("**Pixel size**")
        image_size = st.radio(
            label="$\\hspace{0.1em}\\texttt{Pixel size}$",
            options=("1024x1024", "1792x1024", "1024x1792"),
            # horizontal=True,
            index=0,
            label_visibility="collapsed",
        )

    st.write("")
    st.write("##### Description for your image")

    if st.session_state.image_url is not None:
        st.info(st.session_state.image_description)
        st.image(image=st.session_state.image_url, use_column_width=True)
    
    # Get an image description using the microphone
    audio_bytes = audio_recorder(
        pause_threshold=3.0, text="Speak", icon_size="2x",
        recording_color="#e87070", neutral_color="#6aa36f"        
    )
    if audio_bytes != st.session_state.audio_bytes:
        st.session_state.image_description = read_audio(audio_bytes)
        st.session_state.audio_bytes = audio_bytes
        if st.session_state.image_description is not None:
            st.session_state.prompt_exists = True

    # Get an image description using the keyboard
    text_input = st.chat_input(
        placeholder="Enter a description for your image",
    )
    if text_input:
        st.session_state.image_description = text_input
        st.session_state.prompt_exists = True

    if st.session_state.prompt_exists:
        st.session_state.image_url = chat_image.openai_create_image(
            st.session_state.image_description, model, image_size
        )
        st.session_state.prompt_exists = False
        if st.session_state.image_url is not None:
            st.rerun()

def create_text_image():
    """
    This main function generates text or image by calling
    openai_create_text() or openai_create_image(), respectively.
    """

    st.write("## ChatGPT (RAG)$\,$ &$\,$ DALL·E")

    # Initialize all the session state variables
    initialize.initialize_session_state_variables()

    with st.sidebar:
        st.write("")
        st.write("**API Key Selection**")
        choice_api = st.sidebar.radio(
            label="$\\hspace{0.25em}\\texttt{Choice of API}$",
            options=("Your key", "My key"),
            label_visibility="collapsed",
            horizontal=True,
        )

        if choice_api == "Your key":
            st.write("**Your API Key**")
            st.session_state.openai_api_key = st.text_input(
                label="$\\hspace{0.25em}\\texttt{Your OpenAI API Key}$",
                type="password",
                placeholder="sk-",
                label_visibility="collapsed",
            )
            authen = False if st.session_state.openai_api_key == "" else True
        else:
            st.session_state.openai_api_key = st.secrets["openai_api_key"]
            stored_pin = st.secrets["user_PIN"]
            st.write("**Password**")
            user_pin = st.text_input(
                label="Enter password", type="password", label_visibility="collapsed"
            )
            authen = user_pin == stored_pin

        st.session_state.openai = openai.OpenAI(
            api_key=st.session_state.openai_api_key
        )

        st.write("")
        st.write("**What to Generate**")
        option = st.sidebar.radio(
            label="$\\hspace{0.25em}\\texttt{What to generate}$",
            options=(
                "Text (GPT 3.5)", "Text (GPT 4)",
                "Text with Image", "Image (DALL·E 3)"
            ),
            label_visibility="collapsed",
            # horizontal=True,
            on_change=switch_between_apps,
        )

    if authen:
        if option == "Text (GPT 3.5)":
            create_text("gpt-3.5-turbo-1106")
        elif option == "Text (GPT 4)":
            create_text("gpt-4-1106-preview")
        elif option == "Text with Image":
            chat_image.create_text_with_image("gpt-4-vision-preview")
        else:
            create_image("dall-e-3")
    else:
        st.write("")
        if choice_api == "Your key":
            st.info(
                """
                **Enter your OpenAI API key in the sidebar**

                [Get an OpenAI API key](https://platform.openai.com/api-keys)
                The GPT-4 API can be accessed by those who have made
                a payment of $1 to OpenAI (a strange policy) at the time of
                writing this code.
                """
            )
        else:
            st.info("**Enter the correct password in the sidebar**")

    with st.sidebar:
        st.write("---")
        st.write(
            "<small>**blueholelabs**, Dec. 2023  \n</small>",
            unsafe_allow_html=True,
        )

if __name__ == "__main__":
    create_text_image()