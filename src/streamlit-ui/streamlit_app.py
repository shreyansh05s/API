import requests
import streamlit as st
import pandas as pd
import altair as alt

# Global storage for annotations
if "global_annotations" not in st.session_state:
    st.session_state.global_annotations = pd.DataFrame(columns=["Filename", "Annotation", "Confidence Score"])

# OpenSearch configuration
opensearch_url = "https://opensearch:9200"

# Title with space below it
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Zero-Shot Audio-Annotator</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)  # Line of space between title and file upload box

# Test OpenSearch Connection
try:
    response = requests.get(f"{opensearch_url}/_cluster/health", verify=False, auth=('admin', 'Duck1Teddy#Open'))
except requests.ConnectionError as e:
    st.write(e)
    st.write("Error: Unable to connect to OpenSearch.")

# File upload section
uploaded_file = st.file_uploader("Choose a File (Min. Duration: 10 sec)", type=["wav", "mp3"])
st.markdown("<br><br>", unsafe_allow_html=True)  # Add 2 lines of space after upload functionality

if uploaded_file is not None:
    st.write("Filename:", uploaded_file.name)
    file_content = uploaded_file.read()
    st.audio(file_content, format='audio/wav')

    if st.button("Submit"):
        #st.markdown("<br><br>", unsafe_allow_html=True)  # Add 2 lines of space after submit button

        # Replace 'your_api_url' with the actual API endpoint
        tagging_api_url = "http://tagging:8000/process_audio"
        files = {"file": (uploaded_file.name, file_content)}
        tagging_response = requests.post(tagging_api_url, files=files)
        
        if tagging_response.status_code == 200:
            st.write("Your annotations are ready!")
            st.markdown("<br><br>", unsafe_allow_html=True)
            # Extract annotations and probabilities
            labels = tagging_response.json()["labels"]
            probs = tagging_response.json()["probabilities"]
            relevant_probs = probs[:len(labels)]

            # Create a DataFrame with Filename included
            df = pd.DataFrame({
                "Filename": [uploaded_file.name] * len(labels),  # Add the filename for all rows
                "Annotation": labels,
                "Confidence Score": relevant_probs
            })

            # Update global annotations
            st.session_state.global_annotations = pd.concat([st.session_state.global_annotations, df], ignore_index=True)

            # Adjusted Layout: Add a column of space
            col1, _, col2 = st.columns([1, 0.2, 2])  # Middle column is empty for spacing

            # Column 1: Annotations Table (Exclude Filename)
            with col1:
                #st.subheader("Annotations and Probabilities")
                st.table(df[["Annotation", "Confidence Score"]])  # Exclude Filename from display

                # Add space before the download button
                st.markdown("<br><br>", unsafe_allow_html=True)  # Move Download Tags button down
                # Save DataFrame to CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Annotations as .csv",
                    data=csv,
                    file_name=f"{uploaded_file.name}_tags.csv",
                    mime='text/csv'
                )
                st.markdown("<br><br>", unsafe_allow_html=True)

            # Column 2: Bar Chart
            with col2:
                #st.subheader("Confidence Scores Visualization")
                chart = alt.Chart(df).mark_bar().encode(
                    x='Annotation',
                    y='Confidence Score',
                    color='Annotation', 
                    tooltip=['Annotation', 'Confidence Score']
                ).properties(
                    title='Annotations & Confidence Scores'
                )
                st.altair_chart(chart, use_container_width=True)

        else:
            st.write("Error: Unable to tag the file.")  # Handle tagging API errors
else:
    st.error("No file is provided")  # Handle the initial state before any file is uploaded

# Clustering with Pandas
if not st.session_state.global_annotations.empty:
    #st.subheader("Global Annotation Clusters")

    # Group annotations by their names
    grouped_annotations = st.session_state.global_annotations.groupby("Annotation").agg(
        Mean_Confidence=("Confidence Score", "mean"),
        Total_Occurrences=("Confidence Score", "count")
    ).reset_index()

    # Sort annotations by popularity (occurrences)
    grouped_annotations = grouped_annotations.sort_values(by="Total_Occurrences", ascending=False)

    # Visualize Popular Annotations (Annotations on y-axis, Total Occurrences on x-axis)
    st.markdown("<br><br>", unsafe_allow_html=True)
    bar_chart = alt.Chart(grouped_annotations).mark_bar().encode(
        y=alt.Y('Annotation:N', title='Annotation', sort='-x'),  # Annotations on y-axis
        x=alt.X('Total_Occurrences:Q', title='Total Occurrences'),  # Total Occurrences on x-axis
        color='Annotation:N',
        tooltip=['Annotation', 'Mean_Confidence', 'Total_Occurrences']
    ).properties(
        title="Popular Annotations by Total Occurrences",
        width=800,
        height=400
    )
    st.altair_chart(bar_chart, use_container_width=True)

    # Heatmap Visualization of Annotations and Scores (Annotations on y-axis)
    st.markdown("<br><br>", unsafe_allow_html=True)
    heatmap = alt.Chart(st.session_state.global_annotations).mark_rect().encode(
        y=alt.Y('Annotation:N', title='Annotation'),  # Annotations on y-axis
        x=alt.X('Filename:N', title='Filename'),  # Ensure Filename exists
        color=alt.Color('Confidence Score:Q', scale=alt.Scale(scheme='viridis'), title='Confidence Score'),
        tooltip=['Annotation', 'Filename', 'Confidence Score']
    ).properties(
        title="Annotation Confidence Scores",
        width=800,
        height=400
    )
    st.altair_chart(heatmap, use_container_width=True)
