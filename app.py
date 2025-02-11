import streamlit as st
from PIL import Image
import google.generativeai as genai

# Access the API key from Streamlit secrets
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# Set the title of the app
st.title("🖼️ Math Image Analyzer with Gemini")

# Language selection dropdown
language = st.selectbox(
    "Select the programming language for the generated code:",
    ["Python"]
)

# Upload image
uploaded_image = st.file_uploader("📤 Upload an image (math equations, graphs, etc.)", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    # Display the uploaded image
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Analyze the image using Gemini
    if st.button("Analyze Image"):
        with st.spinner("Analyzing the image..."):
            # Use the newer model (e.g., gemini-1.5-flash)
            model = genai.GenerativeModel('gemini-1.5-flash')

            # Custom prompt based on the selected language
            prompt = f"""
            Analyze this image and provide the mathematical equations, graphs, or related content.
            Also, generate code in {language} to process or visualize the data.
            """

            # Send the image and prompt to Gemini for analysis
            response = model.generate_content([prompt, image])

            # Display the response
            st.subheader("🔍 Analysis Results:")
            st.write(response.text)

            # Extract and display the generated code (if any)
            if "```" in response.text:
                st.subheader(f"🐍 Generated {language} Code:")
                code = response.text.split("```")[1].split("```")[0]
                st.code(code, language=language.lower())