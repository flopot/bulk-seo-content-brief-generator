import streamlit as st
import pandas as pd
import asyncio
import logging
import os
import tiktoken
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)

# Constants
TOKEN_LIMIT = 128000  # GPT-4o-mini context window
SAFETY_MARGIN = 1000  # Leave room for responses
MAX_CONCURRENT_REQUESTS = 5  # Adjust based on rate limits

# Streamlit UI
st.title('Bulk SERP-driven SEO Content Brief Generator')
st.markdown("""by [Florian Potier](https://twitter.com/FloPots) - [Intrepid Digital](https://www.intrepidonline.com/)""", unsafe_allow_html=True)

api_key = st.text_input("Enter your OpenAI API key", type="password")
uploaded_file = st.file_uploader("Choose your CSV file (must contain 'Keyword' and 'URL')", type=['csv'])

if uploaded_file and api_key:
    df = pd.read_csv(uploaded_file)
    required_columns = {"Keyword", "URL"}
    if not required_columns.issubset(df.columns):
        st.error("CSV file must contain 'Keyword' and 'URL' columns.")
        st.stop()

    # OpenAI Async client
    client = AsyncOpenAI(api_key=api_key)
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))

    # Prompt template
    SYSTEM_PROMPT = """
    You'll be given a URL and the keyword it targets. Analyze the page and the top SERP results to generate an optimized content brief.
    """
    USER_PROMPT_TEMPLATE = """
    First, visit {URL}. Then check the top 10 results that appear in the SERP for the keyword '{Keyword}'. Finally, generate a comprehensive content brief respecting the following structure: URL, Primary Keyword, Secondary Keywords, Title, Meta Description, Headings structure (H1, H2, H3 etc... using bullet points), Other comments (here you can add other relevant comments, don't use bullet points or any kind of listing). Here's an example of how it should look like in terms of format:
    
    URL: https://www.ahs.com/our-coverage/
    
    Primary Keyword: Home Warranty Coverage
    
    Secondary Keywords: AHS Home warranty, Home protection plan, Home system coverage, Home appliances warranty, Comprehensive home coverage, Best home warranty coverage, Home warranty plans, Home warranty services.
    
    Title: AHS Home Warranty Coverage: The Comprehensive Home Protection Plan
    
    Meta Description: Secure your property with AHS Home Warranty Coverage. Explore our comprehensive protection plans for your home systems and appliances. Experience peace of mind like never before. 
    
    Headings Structure: 
    - H1: AHS Home Warranty Coverage: Unparalleled Protection for Your Home
    - H2: What is Home Warranty Coverage?
    - H3: The Importance of Home Warranty Coverage
    - H2: Our Comprehensive Home Warranty Plans
    - H3: Home Systems Coverage
    - H3: Home Appliances Warranty
    - H2: How Does AHS Home Warranty Coverage Stand Out?
    - H2: Benefits of Choosing AHS Home Warranty
    - H3: Wide Coverage
    - H3: Affordable and Flexible Plans
    - H3: Professional Services
    - H2: Customer Testimonials
    
    Other comments: Based on the top 10 SERP results, most sites emphasize the nature of their home warranty coverage, the specific systems and appliances that are covered, and their standout features or benefits. However, they neglect to elaborate on the importance of home warranty coverage, and customer testimonials appear lacking as well. This is where your page can stand out from the competition. Also, integrating secondary keywords organically into the content can help improve the page's search engine rankings. Ensure to maintain a user-friendly design and navigation on the page. Include clear call-to-actions (CTAs) to lead visitors towards plan purchase or contacting your team for enquiries. Create high-quality, engaging content to retain visitors and increase dwell time. Remember that content should be written with user intent in mind, rather than just catering to search engine algorithms."""}

    # Async function to generate responses
    async def generate_response(row):
        formatted_prompt = USER_PROMPT_TEMPLATE.format(Keyword=row['Keyword'], URL=row['URL'])
        retry_count = 0
        while retry_count < 5:
            try:
                response = await client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    model="gpt-4o-mini"
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                logging.error(f"Error generating response: {e}")
                retry_count += 1
                await asyncio.sleep(2 ** retry_count)
        return "ERROR"

    # Async batch processing
    async def process_batch(batch_data):
        tasks = [generate_response(row) for row in batch_data]
        return await asyncio.gather(*tasks)

    # Process data asynchronously
    async def process_data():
        progress_bar = st.progress(0)
        response_list = []
        num_rows = len(df)
        processed_rows = 0

        while processed_rows < num_rows:
            batch_data = []
            batch_tokens = 0
            batch_size = 10  # Adjust based on rate limits

            for i in range(batch_size):
                if processed_rows + i >= num_rows:
                    break
                row = df.iloc[processed_rows + i]
                total_prompt_tokens = count_tokens(USER_PROMPT_TEMPLATE.format(Keyword=row['Keyword'], URL=row['URL']))
                batch_tokens += total_prompt_tokens

                if batch_tokens > TOKEN_LIMIT - SAFETY_MARGIN:
                    break
                batch_data.append(row)

            if not batch_data:
                break

            responses = await process_batch(batch_data)
            for row, response in zip(batch_data, responses):
                response_list.append([row['Keyword'], row['URL'], response])

            processed_rows += len(batch_data)
            progress_bar.progress(processed_rows / num_rows)

        return response_list

    if st.button("Generate Content Briefs"):
        results = asyncio.run(process_data())
        results_df = pd.DataFrame(results, columns=['Keyword', 'URL', 'Content Brief'])
        csv_data = results_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Content Briefs as CSV", csv_data, "content-briefs.csv", "text/csv")
