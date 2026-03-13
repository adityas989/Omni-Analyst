import os
import pandas as pd
from dotenv import load_dotenv
from datasets import Dataset

# Ragas 0.1.21 Style Imports
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper

# LangChain Google Integration
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Your Project Imports
from src.agent.agent import OmniAgent

load_dotenv()

class OmniEvaluator:
    def __init__(self, agent: OmniAgent):
        self.agent = agent
        
        # 1. Initialize Gemini Judge (Using 1.5 Pro for grading)
        gemini_llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        self.eval_llm = LangchainLLMWrapper(gemini_llm)

        # 2. Initialize Gemini Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004"
        )

    def run_evaluation(self, test_questions: list):
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }

        # Step 1: Collect Data using the Agent
        for q_data in test_questions:
            question = q_data["question"]
            print(f"🧪 Processing: {question}")
            
            # Use our Agent's logic
            answer, rewritten_query = self.agent.ask(question)
            
            # Retrieve chunks
            raw_results = self.agent.vs.query(rewritten_query, n_results=3)
            contexts = raw_results['documents'][0]

            data["question"].append(question)
            data["answer"].append(answer)
            data["contexts"].append(contexts)
            data["ground_truth"].append(q_data.get("ground_truth", "N/A"))

        # Convert to Dataset
        dataset = Dataset.from_dict(data)
        
        # Step 2: Configure Metrics for 0.1.21
        # In this version, we just pass the LLM/Embeddings to the objects
        metrics = [faithfulness, answer_relevancy, context_precision]
        
        for m in metrics:
            m.llm = self.eval_llm
            if hasattr(m, 'embeddings'):
                m.embeddings = self.embeddings

        # Step 3: Run Evaluation
        print("📊 Running Ragas 0.1.21 evaluation...")
        results = evaluate(
            dataset,
            metrics=metrics
        )
        
        return results.to_pandas()

if __name__ == "__main__":
    my_agent = OmniAgent(db_path="data/vector_db")
    evaluator = OmniEvaluator(my_agent)

    test_set = [
        {
            "question": "What is the primary medical condition described in the patient report?",
            "ground_truth": "The patient is diagnosed with Tuberculosis."
        },
        {
            "question": "Does the document mention any structural failures in the building design?",
            "ground_truth": "No structural failures were reported."
        }
    ]

    report = evaluator.run_evaluation(test_set)
    
    print("\n" + "="*60)
    print("📈 OMNI-ANALYST QUALITY REPORT (Ragas 0.1.21)")
    print("="*60)
    print(report[['question', 'faithfulness', 'answer_relevancy', 'context_precision']])