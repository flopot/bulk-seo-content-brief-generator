import streamlit as st
import pandas as pd
from openai import OpenAI
import time
import concurrent.futures

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
uploaded_file = st.file_uploader(
    "Choose your CSV file. It should contain the columns 'Keyword' and 'URL' (See [Template](https://docs.google.com/spreadsheets/d/1ApdoOKjC6ZAg1JkWiY51fOToXqH_cTP8PSj2wVlS8Iw/edit?usp=sharing))",
    type=['csv']
)

if uploaded_file and api_key:
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Read the uploaded file into a DataFrame
    keywords_df = pd.read_csv(uploaded_file)
    
    # Initialize progress bar
    progress_bar = st.progress(0)
    total_keywords = len(keywords_df)

    # Function to generate SEO recommendations using the OpenAI client
    def generate_seo_recommendations(batch):
        messages = [{"role": "system", "content": 
            "You will receive a list of URLs and keywords. For each, analyze the page and top 10 SERP results. "
            "Return a structured content brief for each URL, using concise formatting."}
        ]

        for keyword, url in batch:
            messages.append({"role": "user", "content": f"Keyword: {keyword}\nURL: {url}\nProvide the structured content brief."})

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            return [choice.message.content.strip() for choice in response.choices]
        except Exception as e:
            return [f"Error: {e}" for _ in batch]

    # **Batching for faster execution**
    batch_size = 5  # Adjust based on token limits
    all_responses = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for i in range(0, total_keywords, batch_size):
            batch = keywords_df.iloc[i:i+batch_size][['Keyword', 'URL']].values.tolist()
            futures.append(executor.submit(generate_seo_recommendations, batch))
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            responses = future.result()
            all_responses.extend(responses)
            progress_bar.progress(min((i+1) * batch_size / total_keywords, 1.0))

    # Convert the responses into a DataFrame
    results_df = pd.DataFrame(zip(keywords_df['Keyword'], all_responses), columns=['Keyword', 'Recommendations'])

    # Convert DataFrame to CSV and create download button
    csv = results_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Content Briefs as CSV", csv, "content-briefs.csv", "text/csv")

