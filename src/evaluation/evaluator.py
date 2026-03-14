import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Ragas Imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# LangChain Google Integration
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Your Project Imports
from src.agent.agent import OmniAgent

load_dotenv()

# --- THE MONKEY PATCH ---
# This class "intercepts" the call from Ragas and removes 'temperature'
class SafeGeminiWrapper(LangchainLLMWrapper):
    def generate(self, *args, **kwargs):
        # Ragas 0.1.21 injects 'temperature' into kwargs which breaks Gemini 2.x/1.5
        if 'temperature' in kwargs:
            del kwargs['temperature']
        return super().generate(*args, **kwargs)

    async def agenerate(self, *args, **kwargs):
        if 'temperature' in kwargs:
            del kwargs['temperature']
        return await super().agenerate(*args, **kwargs)
# ------------------------

class OmniEvaluator:
    def __init__(self, agent: OmniAgent):
        self.agent = agent
        
        # Initialize Gemini Judge
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        gemini_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # USE THE PATCHED WRAPPER HERE
        self.eval_llm = SafeGeminiWrapper(langchain_llm=gemini_llm)
        self.eval_embeddings = LangchainEmbeddingsWrapper(gemini_embeddings)

    def run_evaluation(self, test_questions: list):
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        for q_data in test_questions:
            question = q_data["question"]
            print(f"🧪 Processing: {question}")
            
            answer, rewritten_query = self.agent.ask(question)
            raw_results = self.agent.vs.query(rewritten_query, n_results=3)
            contexts = [str(doc) for doc in raw_results['documents'][0]]

            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(q_data.get("ground_truth", "N/A"))

        dataset = Dataset.from_dict(data)
        
        metrics = [faithfulness, answer_relevancy, context_precision]
        for m in metrics:
            m.llm = self.eval_llm
            if hasattr(m, 'embeddings'):
                m.embeddings = self.eval_embeddings

        print("📊 Running Patched Ragas evaluation (Sequential)...")
        # is_async=False is mandatory to ensure the patch works reliably
        results = evaluate(
            dataset,
            metrics=metrics
        )
        
        return results.to_pandas()

if __name__ == "__main__":
    my_agent = OmniAgent(db_path="data/vector_db")
    evaluator = OmniEvaluator(my_agent)

    test_set = [
        {"question": "What is the medical condition?", "ground_truth": "Tuberculosis"},
        {"question": "Are there structural failures?", "ground_truth": "No"}
    ]

    report = evaluator.run_evaluation(test_set)
    print("\n" + "="*60)
    print("📈 FINAL PATCHED REPORT")
    print("="*60)
    print(report)