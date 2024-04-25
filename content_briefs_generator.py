import streamlit as st
import pandas as pd
from openai import OpenAI
import time

# Title and Setup
st.title('Bulk SERP-driven SEO Content Brief Generator')

# Subtitle
st.markdown(
    """
    by [Florian Potier](https://twitter.com/FloPots) - [Intrepid Digital](https://www.intrepidonline.com/)
    """,
    unsafe_allow_html=True
)

# Input for the OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")

# File upload
uploaded_file = st.file_uploader("Choose your CSV file", type=['csv'])

if uploaded_file and api_key:
    # Initialize the OpenAI client with the user-provided API key
    client = OpenAI(api_key=api_key)  # Use the input API key here

    # Read the uploaded file into a DataFrame
    keywords_df = pd.read_csv(uploaded_file)
    
    all_responses = []

    # Function to generate SEO recommendations using the OpenAI client
    def generate_seo_recommendations(keyword, url):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Send your URL and the keyword you target with it. I will have a look at your page and at the top 10 results to generate the best possible content brief."},
                {"role": "user", "content": f"First, visit '{url}'. Then check the top 10 results that appear in the SERP for the keyword'{keyword}'. Finally, generate a comprehensive content brief respecting the following structure: URL, Primary Keyword, Secondary Keywords, Title, Meta Description, Headings structure (H1, H2, H3 etc... using bullet points), Other comments (here you can add other relevant comments, don't use bullet points or any kind of listing)."}
            ],
            model="gpt-4"
        )
        # Correctly access the message content from the response
        return response.choices[0].message.content.strip()

    # Iterate over each row in the DataFrame
    for index, row in keywords_df.iterrows():
        seo_advice = generate_seo_recommendations(row['Keyword'], row['URL'])
        all_responses.append([row['Keyword'], seo_advice])
        time.sleep(1)  # To avoid hitting API rate limits

    # Convert the responses into a DataFrame
    results_df = pd.DataFrame(all_responses, columns=['Keyword', 'Recommendations'])

    # Convert DataFrame to CSV and create download button
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Content Briefs as CSV", csv, "content-briefs.csv", "text/csv")
