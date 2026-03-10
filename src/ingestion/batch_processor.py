import json
import os
from pathlib import Path
from tqdm import tqdm  # For progress bars
from src.ingestion.processor import PDFProcessor
from src.ingestion.captioner import ImageCaptioner

class BatchProcessor:
    def __init__(self, input_dir: str, output_file: str):
        self.input_dir = Path(input_dir)
        self.output_file = Path(output_file)
        
        # Load the captioning model ONCE (Heavy operation)
        self.captioner = ImageCaptioner() 
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

    def generate_table_summary(self, table_md: str, source_name: str) -> str:
        """
        Creates a semantic 'hook' for tables. 
        In a full version, you could use an LLM here.
        """
        # Extract the first line (headers) to create a summary
        lines = table_md.split('\n')
        headers = lines[0] if len(lines) > 0 else "Unknown data"
        return f"Data table from {source_name} containing columns: {headers}"

    def process_all(self):
        pdf_files = list(self.input_dir.glob("*.pdf"))
        if not pdf_files:
            print("❌ No PDF files found in data/raw!")
            return

        print(f"📂 Found {len(pdf_files)} PDFs. Starting batch processing...")

        # Open JSONL for streaming
        with open(self.output_file, "w", encoding="utf-8") as f:
            # Using tqdm for a professional progress bar
            for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
                try:
                    processor = PDFProcessor(pdf_path)
                    docs, images = processor.process_file()
                    
                    # --- 1. ENRICH IMAGES (Multimodal) ---
                    for img in images:
                        caption = self.captioner.get_caption(img['path'])
                        if caption:
                            img['caption'] = caption
                            # Inject caption into searchable documents
                            docs.append({
                                "content": f"Image (Visual Findings): {caption}",
                                "metadata": {
                                    "source": img['source'],
                                    "page": img['page'],
                                    "type": "image_caption",
                                    "image_path": img['path']
                                }
                            })

                    # --- 2. ENRICH TABLES (Semantic Hooks) ---
                    for doc in docs:
                        if doc['metadata']['type'] == 'table':
                            summary = self.generate_table_summary(doc['content'], pdf_path.name)
                            # Prepend the summary to the markdown table content
                            doc['content'] = f"Table Summary: {summary}\n\n{doc['content']}"

                    # --- 3. WRITE TO DISK (JSONL) ---
                    file_entry = {
                        "file_name": pdf_path.name,
                        "documents": docs
                    }
                    f.write(json.dumps(file_entry) + "\n")

                except Exception as e:
                    print(f"\n❌ Error in {pdf_path.name}: {e}")

        print(f"\n🚀 Ingestion Complete! Master Index: {self.output_file}")

if __name__ == "__main__":
    bp = BatchProcessor(
        input_dir="data/raw",
        output_file="data/processed/master_index.jsonl"
    )
    bp.process_all()