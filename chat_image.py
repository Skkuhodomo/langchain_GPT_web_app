import streamlit as st
import requests
import base64, requests
from io import BytesIO
def openai_create_image(description, model="dall-e-3", size="1024x1024"):
    """
    This function generates image based on user description.

    Args:
        description (string): User description
        model (string): Default set to "dall-e-3"
        size (string): Pixel size of the generated image

    Return:
        URL of the generated image
    """

    try:
        with st.spinner("AI is generating..."):
            response = st.session_state.openai.images.generate(
                model=model,
                prompt=description,
                size=size,
                quality="standard",
                n=1,
            )
        image_url = response.data[0].url
    except Exception as e:
        image_url = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return image_url


def openai_query_image_url(image_url, query, model="gpt-4-vision-preview"):
    """
    This function answers the user's query about the given image from a URL.

    Args:
        image_url (string): URL of the image
        query (string): the user's query
        model (string): default set to "gpt-4-vision-preview"

    Return:
        text as an answer to the user's query.
    """

    try:
        with st.spinner("AI is thinking..."):
            response = st.session_state.openai.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{query}"},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"{image_url}"},
                            },
                        ],
                    },
                ],
                max_tokens=300,
            )
        generated_text = response.choices[0].message.content
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text


def openai_query_uploaded_image(image_b64, query, model="gpt-4-vision-preview"):
    """
    This function answers the user's query about the uploaded image.

    Args:
        image_b64 (base64 encoded string): base64 encoded image
        query (string): the user's query
        model (string): default set to "gpt-4-vision-preview"

    Return:
        text as an answer to the user's query.
    """

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {st.session_state.openai_api_key}"
    }

    payload = {
        "model": f"{model}",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{query}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        with st.spinner("AI is thinking..."):
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
        generated_text = response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        generated_text = None
        st.error(f"An error occurred: {e}", icon="🚨")

    return generated_text

def image_to_base64(image):
    """
    This function converts an image object from PIL to a base64
    encoded image, and returns the resulting encoded image.
    """

    # Convert the image to RGB mode if necessary
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Save the image to a BytesIO object
    buffered_image = BytesIO()
    image.save(buffered_image, format="JPEG")

    # Convert BytesIO to bytes and encode to base64
    img_str = base64.b64encode(buffered_image.getvalue())

    # Convert bytes to string
    base64_image = img_str.decode("utf-8")

    return base64_image


def shorten_image(image, max_pixels=1024):
    """
    This function takes an Image object as input, and shortens the image size
    if the image is greater than max_pixels x max_pixels.
    """

    if max(image.width, image.height) > max_pixels:
        if image.width > image.height:
            new_width, new_height = 1024, image.height * 1024 // image.width
        else:
            new_width, new_height = image.width * 1024 // image.height, 1024

        image = image.resize((new_width, new_height))

    return image

