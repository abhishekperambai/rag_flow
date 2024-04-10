from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np, pandas as pd
import pathlib


class VectorIndex:
    def __init__(self, data, chunk_flag=False):
        self.data = data
        self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.chunk_flag = chunk_flag

    def get_content_index(self):
        """
        Based on chunk flag retreive the content along with chunk number of page number
        Here, the PDF document had numbers printed at top of a page, which ended up with 115
        and we treated this as a chunk number. Feel free to modify the code according to your logic.
        """
        df = pd.read_csv(self.data)
        
        if self.chunk_flag:
            df = df[df["chunk"]!=0]
            content = df["chunk"].astype(str) + " " + df["content"]
        else:
            content = df["page_num"].astype(str) + " " + df["content"]
        return [str(text) for text in content.values] 

    def generate_embeddings(self, sentences):
        """
        Returns sentence embeddings based on the model initialized in the init func
        """
        return self.model.encode(sentences)

    def create_index(self):
        """
        Generates the index with FAISS library and persists the file 
        under the output directory in root.
        """
        content = self.get_content_index()
        embeddings = self.generate_embeddings(content)
        faiss_model = faiss.IndexFlatL2(embeddings.shape[1])
        faiss_model.add(embeddings)
        if self.chunk_flag:
            faiss.write_index(faiss_model, "./output/faiss_index_multiqa_chunks.index")
        else:
            faiss.write_index(faiss_model, "./output/faiss_index_multiqa.index")

if __name__=="__main__":
    if not pathlib.Path(".output/faiss_index_multiqa.index").exists():
        obj = VectorIndex("./output/extracted_df_processed.csv", chunk_flag=False)
        obj.create_index()
    else:
        print("Index Already Exists and Generated")



























# # from langchain_core.messages import HumanMessage
# # from langchain_mistralai.chat_models import ChatMistralAI

# # chat = ChatMistralAI(mistral_api_key=mistral_api_key)
# # messages = [HumanMessage(content="knock knock")]

# # print(chat.invoke(messages))


# # Import the required libraries
# from mistralai.client import MistralClient
# from mistralai.models.chat_completion import ChatMessage

# # Create an api client
# client = MistralClient(api_key=mistral_api_key)

# # Create the prompts
# messages = [
#     ChatMessage(role="system", content="You should act as an AI assistant to answer the query of the user based on the context provided to you."),
#     ChatMessage(role="user", content="""<>""")
# ]

# # Call the chat api
# chat_response = client.chat(
#     model="mistral-large-latest",
#     messages=messages,
# )

# # Print the response
# print(chat_response.choices[0].message.content)



