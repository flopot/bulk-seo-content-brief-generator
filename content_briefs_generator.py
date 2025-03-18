import streamlit as st
import pandas as pd
from openai import OpenAI
import time
import concurrent.futures

# Title and Setup
st.title('Bulk SEO Content Brief Generator')

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
            "You will receive a URL and the keyword it targets. Use these to generate the best possible content brief. Don't say anything more than the content brief."}
        ]

        for keyword, url in batch:
            messages.append({"role": "user", "content":f"""Here's the URL: '{url}' and the Keyword: '{keyword}'. Generate a comprehensive content brief respecting the following structure: URL, Primary Keyword, Secondary Keywords, Title, Meta Description, Headings structure (H1, H2, H3 etc... using bullet points), Other comments (here you can add other relevant comments, don't use bullet points or any kind of listing). Here's an example of how it should look like in terms of format:
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
    
    Other comments: Based on the top 10 SERP results, most sites emphasize the nature of their home warranty coverage, the specific systems and appliances that are covered, and their standout features or benefits. However, they neglect to elaborate on the importance of home warranty coverage, and customer testimonials appear lacking as well. This is where your page can stand out from the competition. Also, integrating secondary keywords organically into the content can help improve the page's search engine rankings. Ensure to maintain a user-friendly design and navigation on the page. Include clear call-to-actions (CTAs) to lead visitors towards plan purchase or contacting your team for enquiries. Create high-quality, engaging content to retain visitors and increase dwell time. Remember that content should be written with user intent in mind, rather than just catering to search engine algorithms."""})

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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

