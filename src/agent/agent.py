import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
from src.retrieval.vector_store import VectorStore

load_dotenv()

class OmniAgent:
    def __init__(self, db_path: str):
        self.vs = VectorStore(db_path=db_path)
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_id = "gemini-2.5-flash"
        
        self.history = [] 

    def _rewrite_query(self, user_query: str):
        """Converts ambiguous follow-ups into standalone search queries."""
        if not self.history:
            return user_query 

        history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.history[-3:]])

        rewrite_prompt = f"""
        Given the following chat history and a new question, rewrite the question to be a 
        standalone search query that contains all necessary context (people, diseases, dates, etc.).
        
        History:
        {history_str}
        
        New Question: {user_query}
        
        Standalone Query:"""

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=rewrite_prompt
        )
        return response.text.strip()

    def ask(self, user_query: str):
        try:
            standalone_query = self._rewrite_query(user_query)
            print(f"🔄 Rewritten Query: {standalone_query}")

            results = self.vs.query(standalone_query, n_results=5)
            context_text = "\n".join([f"[{m.get('source_file')}] {t}" for t, m in zip(results['documents'][0], results['metadatas'][0])])

            final_prompt = f"""
            You are the 'Omni-Analyst'. Your role is to analyze technical documents.
            Rules:
            1. If the answer is not in the context, say "I don't have enough information in the documents to answer this."
            2. Always cite your sources in your answer (e.g., "[Source: file.pdf, Page 3]").
            3. If you use information from an 'image_caption', mention that it was found in an image.
            4. Keep your tone professional and objective.
            Context: {context_text}
            Question: {user_query}
            """
            
            response = self.client.models.generate_content(model=self.model_id, contents=final_prompt)
            answer = response.text

            self.history.append({"role": "user", "content": user_query})
            self.history.append({"role": "assistant", "content": answer})

            return answer, standalone_query 
            
        except Exception as e:
            return f"❌ Agent Error: {e}", user_query
        
        
if __name__ == "__main__":
    agent = OmniAgent(db_path="data/vector_db")
    print(agent.ask("What is the primary finding in the medical reports?"))
