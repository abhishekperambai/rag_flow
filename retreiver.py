from sentence_transformers import SentenceTransformer, CrossEncoder
from indexing import VectorIndex
import faiss, torch
import numpy as np, pandas as pd
import pathlib
# from llmlingua import PromptCompressor
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class Retriever(VectorIndex):
    def __init__(self, query, extracted_df_fp, index_fp, chunk_flag=False):
        self.query = query
        self.chunk_flag = chunk_flag
        self.model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        # self.compressor = PromptCompressor(device_map="cpu")
                                        #     model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                                        #     use_llmlingua2=True,
                                        # )
        self.extracted_df_fp = extracted_df_fp
        self.index_fp = index_fp
        self.init_k = 10
        self.final_k = 3

    
    def read_extracted_df(self):
        """
        Imports the extracted dataframe from the document parser step
        """
        return pd.read_csv(self.extracted_df_fp)

    
    def get_top_init_k(self, df, index_ids):
        """
        Iterate through the extracted dataframe from doc parser step and filter out the
        index ID's with top scores. Depending upon the chunk flag, the code triggers the 
        approproate logic.
        """
        seen_candidates = set()
        retrieval_counter, index =0, 0
        top_k = 5
        output = []
        for _, row in df.iloc[index_ids].iterrows():
                if self.chunk_flag:
                    if row["chunk"] not in seen_candidates and retrieval_counter < self.init_k:
                        output.append(
                            {
                                "chunk": row["chunk"],
                                "content": row["content"],
                                "index_id": index_ids[index],
                            }
                        )
                        retrieval_counter += 1
                    index += 1
                    seen_candidates.add(row["chunk"])
                else:
                    if row["page_num"] not in seen_candidates and retrieval_counter < self.init_k:
                        output.append(
                            {
                                "page_num": row["page_num"],
                                "content": row["content"],
                                "index_id": index_ids[index],
                            }
                        )
                        retrieval_counter += 1
                    index += 1
                    seen_candidates.add(row["page_num"])
        return output
    
    def re_rank(self, output):
        """
        Leverages cross encoder model to rerank the retreival results 
        """
        if self.chunk_flag:
            cross_encoder_ip = [(self.query, str(entry["chunk"]) + " " + entry["content"]) for entry in output]
        else:
            cross_encoder_ip = [(self.query, str(entry["page_num"]) + " " + entry["content"]) for entry in output]
        re_rank_scores = self.cross_encoder.predict(cross_encoder_ip).tolist()
        for idx in range(len(re_rank_scores)):
            output[idx][
                "cross_encoder_score"
            ] = re_rank_scores[idx]

        re_ranked_output = sorted(
                                    output,
                                    key=lambda x: x["cross_encoder_score"],
                                    reverse=True,
                                )
        return re_ranked_output
    
    def search_index(self, emb):
        """
        Searches the vector index wrt the query emb and returns similarity scores and
        index ID's
        """
        faiss_index = faiss.read_index(self.index_fp)
        sim_scores, index_ids = faiss_index.search(np.array(
                                                   [[value for value in emb[0]]])
                                                   , 10)
        return sim_scores.tolist()[0], index_ids.tolist()[0]
    
    
    def post_process(self, output):
        """
        Post process the text by appending top 3 matches into a single string,
        to provide it as a context to the LLM 
        """
        # temp = pd.DataFrame(output[:self.final_k])
        return "\n\n".join([out["content"] for out in output[:3]])
    
    def compress_context(self, output, query):
        """
        This func leverages LLMLingua model to compress the prompt based on the 
        instruction provided and the question from the user
        """
        instruct = """Based on the given context, answer the user query and \
                        also highlight the text which has the answer from the given context"""
        return self.compressor.compress_prompt(output, 
                                               instruction=instruct, 
                                               question=query, 
                                               target_token=1800,
                                               force_tokens = ['\n', '?'])


    
    def retrieve(self):
        """
        Triggers the retreival process from the  vector index
        """
        query_embeddings = self.generate_embeddings([self.query]) # inherited method from VectorIndex class
        sim_scores, index_ids = self.search_index(query_embeddings)
        df = self.read_extracted_df()
        results = self.get_top_init_k(df, index_ids)
        re_ranked_results = self.re_rank(results)
        output = self.post_process(re_ranked_results)
        return output
        # final_out = self.compress_context(output, self.query)
        # return final_out


        
if __name__ == "__main__":
    query = "Which place was Nelson Mandela born?"
    extracted_df_fp = "./output/extracted_df_processed.csv"
    index_fp = "./output/faiss_index_multiqa.index"
    retreiver = Retriever(query, extracted_df_fp, index_fp)
    context = retreiver.retrieve()
    print("#"*45)
    print("Context-------", context)