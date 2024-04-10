# RAG Based Approach - Integration with Mistral AI LLM for efficient response on internal documents

## Steps to run the codebase

1. Create a python virtual environment and install the dependencies from requirements.txt file.
2. Place the PDF under datasets directory in root.
3. Run the document_parser.py file to extract the text from PDF. Change the **parameters in this codebase if images exist in the PDF file**.
4. This will generate two csv files under the **output directory** in root (**extracted_df.csv and extracted_df_processed.csv**).
5. **extracted_df_processed.csv** has the appropriate post processing methods applied to improve the semantic search mechanism.
6. Run the **indexing.py** file to generate the index, this will generate the index file under output directory in root.
7. Now run **app.py** file to trigger the gradio frontend to interact with **MistralAI LLM** with questions specific to the document under dataset directory.

### Note:
Change and add your MistralAI key under environment variables with name "MISTRAL_AI_API_KEY" for the code to work seamlessly.
