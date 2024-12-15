import requests
import streamlit as st
import pandas as pd
import altair as alt

# OpenSearch configuration
opensearch_url = "https://opensearch:9200"

# Label Studio configuration
# label_studio_url = "http://label-studio:8080"

st.title("Zero-Shot Annotator")

# Test OpenSearch Connection
try:
    response = requests.get(f"{opensearch_url}/_cluster/health", verify=False, auth=('admin', 'Duck1Teddy#Open'))
    #st.write("OpenSearch Status Code:", response.status_code)
    #st.write("OpenSearch Headers:", response.content)
    # st.write("OpenSearch Status:", response.json())
except requests.ConnectionError as e:
    st.write(e)
    st.write("Error: Unable to connect to OpenSearch.")

# Test Label Studio Connection
# try:
#     response = requests.get(f"{label_studio_url}/api/projects", headers={"Authorization": "Token YOUR_LABEL_STUDIO_API_KEY"})
#     st.write("Label Studio Projects:", response.json())
# except requests.ConnectionError:
#     st.write("Error: Unable to connect to Label Studio.")

# File upload section
uploaded_file = st.file_uploader("Choose a File (Min. Duration: 10 sec)", type=["wav", "mp3"])
tagging_response = None
embedding_response = None
if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
     # Display file content
    file_content = uploaded_file.read()
    st.audio(file_content, format='audio/wav')

    # Submit button
    if st.button("Submit"):
        # Replace 'your_api_url' with the actual API endpoint
        tagging_api_url = "http://tagging:8000/process_audio"
        # embedding_api_url = "http://your-api-service/upload"

        # Send file to the API
        files = {"file": (uploaded_file.name, file_content)}
        # data = {"audio_name": uploaded_file.name}
        tagging_response = requests.post(tagging_api_url, files=files)
        # embedding_response = requests.post(embedding_api_url, files=files, data={"filename": uploaded_file.name})
        
       # st.write("Tagging Status Code:", tagging_response.status_code)
        # st.write("Tagging response:", tagging_response.content)
        
        if tagging_response.status_code == 200:
            st.write("Your annotations are ready!")
            # show the tags
            labels = tagging_response.json()["labels"]	
            probs = tagging_response.json()["probabilities"]

            # Filter probabilities to match the length of identified tags
            relevant_probs = probs[:len(labels)]

            #st.write("Annotations:", labels)
            #st.write("Probabilities:", relevant_probs)

            # Create a DataFrame
            df = pd.DataFrame({"Annotation": labels, "Confidence Score": relevant_probs})

            # Display the DataFrame
            st.dataframe(df)

            # Save DataFrame to CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Tags as CSV",
                data=csv,
                file_name=f"{uploaded_file.name}_tags.csv",
                mime='text/csv'
            )

            # Visualization 
            chart = alt.Chart(df).mark_bar().encode( 
                x='Annotation', 
                y='Confidence Score', 
                color='Annotation', 
                tooltip=['Annotation', 'Confidence Score'] 
            ).properties( 
                title='Annotation & Confidence Scores' 
            ) 
            st.altair_chart(chart, use_container_width=True)
            
        else:
            st.write("Error: Unable to tag the file.")
else:
    st.error("No file is provided")
