import os
from google import genai
from google.genai import types # For advanced configurations
from dotenv import load_dotenv
from src.retrieval.vector_store import VectorStore

load_dotenv()

class OmniAgent:
    def __init__(self, db_path: str):
        self.vs = VectorStore(db_path=db_path)
        
        # New SDK pattern: Initialize a Client
        # It automatically looks for GOOGLE_API_KEY in your environment
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = "gemini-2.5-flash"

    def ask(self, query: str):
        try:
            # 1. Retrieve context from your Vector DB
            results = self.vs.query(query, n_results=5)
            
            # 2. Build the system prompt
            prompt = self._build_prompt(query, results)
            
            # 3. Generate content using the NEW client method
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            return f"❌ Agent Error: {e}"
        
    def _build_prompt(self, query: str, context_chunks: list):
        """Constructs a grounded prompt for the LLM."""
        context_text = "\n\n".join([
            f"--- Context Block ---\nSource: {meta.get('source_file')}\nPage: {meta.get('page')}\nType: {meta.get('type')}\nContent: {text}"
            for text, meta in zip(context_chunks['documents'][0], context_chunks['metadatas'][0])
        ])

        system_prompt = f"""
        You are the 'Omni-Analyst', a professional AI assistant. 
        Your task is to answer the user's question based ONLY on the provided context blocks.
        
        Rules:
        1. If the answer is not in the context, say "I don't have enough information in the documents to answer this."
        2. Always cite your sources in your answer (e.g., "[Source: file.pdf, Page 3]").
        3. If you use information from an 'image_caption', mention that it was found in an image.
        4. Keep your tone professional and objective.

        Context:
        {context_text}

        User Question: {query}
        """
        return system_prompt

    # def _build_prompt(self, query, context_chunks):
    #     # ... (Your existing prompt building logic remains exactly the same) ...
    #     context_text = "\n\n".join([
    #         f"--- Context Block ---\nSource: {meta.get('source_file')}\nPage: {meta.get('page')}\nContent: {text}"
    #         for text, meta in zip(context_chunks['documents'][0], context_chunks['metadatas'][0])
    #     ])

    #     return f"""
    #     You are the 'Omni-Analyst'. Answer based ONLY on the context below.
        
    #     Context:
    #     {context_text}

    #     Question: {query}
    #     """

if __name__ == "__main__":
    agent = OmniAgent(db_path="data/vector_db")
    print(agent.ask("What is the primary finding in the medical reports?"))
