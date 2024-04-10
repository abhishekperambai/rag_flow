# Import the required libraries
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import os


class LLM:
    def __init__(self, query, context):
        self.query = query
        self.context = context
        mistral_api_key = os.environ.get("MISTRAL_AI_API_KEY")
        self.client = MistralClient(api_key=mistral_api_key)
    
    def answer_query(self):
        user_prompt = f"""Based on the given context, answer the user query and \
                        also highlight the text which has the answer from the given context \
                        
                        #### CONTEXT ##### \
                        {self.context} \
                            
                        #### QUERY #### \
                        Query: {self.query} """
        messages = [
                      ChatMessage(role="system", content="You should act as an AI assistant to answer the query of the user based on the context provided to you."),
                      ChatMessage(role="user", content=user_prompt)
                   ]

        chat_response = self.client.chat(
                                        model="mistral-large-latest",
                                        messages=messages,
                                    )
        
        return chat_response.choices[0].message.content