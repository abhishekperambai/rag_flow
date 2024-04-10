import gradio as gr
from retreiver import Retriever
from llm import LLM

extracted_df_fp = "./output/extracted_df_processed.csv"
index_fp = "./output/faiss_index_multiqa.index"

def run_app(query, history):
    retreiver = Retriever(query, extracted_df_fp, index_fp)
    context = retreiver.retrieve()
    print("#"*30)
    print("CONTEXT\n\n",context)

    print("#"*30)
    print("LENGTH",len(context))
    llm = LLM(query, context)
    return llm.answer_query()

demo = gr.ChatInterface(fn=run_app, title="RAG Framework Integration with LLM")
demo.launch()