import streamlit as st
from PIL import Image
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI
import os
import requests  # For direct Groq API calls

# Set the title of the app
st.title("🖼️ Latex Code Generator")

# Model selection dropdown
model_choice = st.selectbox(
    "Select the AI model for analysis:",
    ["Gemini", "OpenAI", "Groq"]
)

# API key input based on model selection
api_key = None
if model_choice == "OpenAI":
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
elif model_choice == "Groq":
    api_key = st.text_input("Enter your Groq API key:", type="password")
elif model_choice == "Gemini":
    api_key = st.secrets["GOOGLE_API_KEY"]  # Fetch from Streamlit secrets

# Upload image
uploaded_image = st.file_uploader("📤 Upload an image (math equations, graphs, topology, etc.)", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Analyze the image
    if st.button("Analyze Image"):
        if not api_key:
            st.error("Please provide a valid API key for the selected model.")
        else:
            with st.spinner("Analyzing the image..."):
                try:
                    # Initialize the selected model
                    if model_choice == "Gemini":
                        genai.configure(api_key=api_key)
                        model = genai.GenerativeModel('gemini-1.5-flash')
                        response = model.generate_content([
                            "Analyze this image and provide the mathematical equations, graphs, or topology details. Also, generate LaTeX code for the content.",
                            image
                        ])
                        result = response.text

                    elif model_choice == "OpenAI":
                        llm = OpenAI(api_key=api_key)
                        prompt = PromptTemplate(
                            input_variables=["image_description"],
                            template="Analyze this image description and provide the mathematical equations, graphs, or topology details. Also, generate LaTeX code for the content. Image Description: {image_description}"
                        )
                        chain = LLMChain(llm=llm, prompt=prompt)
                        result = chain.run(image_description="Math image uploaded by the user.")

                    elif model_choice == "Groq":
                        # Use Groq API directly
                        headers = {
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json"
                        }
                        data = {
                            "prompt": "Analyze this image description and provide the mathematical equations, graphs, or topology details. Also, generate LaTeX code for the content. Image Description: Math image uploaded by the user.",
                            "max_tokens": 500
                        }
                        response = requests.post(
                            "https://api.groq.com/v1/completions",  # Replace with the correct Groq API endpoint
                            headers=headers,
                            json=data
                        )
                        if response.status_code == 200:
                            result = response.json()["choices"][0]["text"]
                        else:
                            st.error(f"Groq API error: {response.status_code} - {response.text}")
                            result = None

                    # Display the result
                    if result:
                        st.subheader("🔍 Analysis Results:")
                        st.write(result)

                        # Extract and display LaTeX code (if any)
                        if "```latex" in result:
                            st.subheader("📜 LaTeX Code:")
                            latex_code = result.split("```latex")[1].split("```")[0]
                            st.code(latex_code, language="latex")

                except Exception as e:
                    st.error(f"An error occurred: {e}")