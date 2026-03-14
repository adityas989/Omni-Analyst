from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import pdfplumber
import os
import re
from pathlib import Path

class PDFProcessor:
    def __init__(self, file_path: str, output_image_dir: str = "data/processed/images"):
        self.file_path = Path(file_path)
        self.output_image_dir = Path(output_image_dir)
        self.output_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Big Tech Standard: Small chunks with overlap and specific metadata
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )

    def clean_text(self, text: str) -> str:
        """Issue #2: Removing broken line breaks and normalizing whitespace."""
        text = text.replace("\n", " ")
        text = re.sub(r'\s+', ' ', text)  # Removes multiple spaces/tabs
        return text.strip()

    def format_table_to_markdown(self, table) -> str:
        """Converts raw list tables to Markdown for better LLM reasoning."""
        if not table or not table[0]: return ""
        clean_table = [[str(item) if item is not None else "" for item in row] for row in table]
        headers = clean_table[0]
        rows = clean_table[1:]
        
        md_table = f"| {' | '.join(headers)} |\n"
        md_table += f"| {' | '.join(['---'] * len(headers))} |\n"
        for row in rows:
            md_table += f"| {' | '.join(row)} |\n"
        return md_table

    def process_file(self):
        """Issue #9: Optimized Pipeline Flow."""
        final_documents = []
        image_metadata = []

        # 1. Open Document
        with fitz.open(self.file_path) as doc:
            file_name = self.file_path.name
            
            # Process Page by Page for granular metadata (Issue #5)
            for page_index, page in enumerate(doc):
                # --- TEXT PROCESSING ---
                raw_text = page.get_text("text")
                cleaned_text = self.clean_text(raw_text)
                
                if cleaned_text:
                    chunks = self.text_splitter.split_text(cleaned_text)
                    for chunk in chunks:
                        final_documents.append({
                            "content": chunk,
                            "metadata": {
                                "source": file_name,
                                "page": page_index + 1,
                                "type": "text"
                            }
                        })

                # --- IMAGE PROCESSING (Issue #4 & #7) ---
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    
                    # Save to Disk immediately (Memory Safety)
                    img_ext = base_image["ext"]
                    img_filename = f"{self.file_path.stem}_p{page_index+1}_i{img_index}.{img_ext}"
                    img_path = self.output_image_dir / img_filename
                    
                    with open(img_path, "wb") as f:
                        f.write(base_image["image"])
                    
                    image_metadata.append({
                        "type": "image",
                        "page": page_index + 1,
                        "path": str(img_path),
                        "source": file_name
                    })

        # --- TABLE PROCESSING (Issue #6) ---
        with pdfplumber.open(self.file_path) as pdf:
            for page_index, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    md_table = self.format_table_to_markdown(table)
                    if md_table:
                        # Chunk the table if it's too large
                        table_chunks = self.text_splitter.split_text(md_table)
                        for t_chunk in table_chunks:
                            final_documents.append({
                                "content": t_chunk,
                                "metadata": {
                                    "source": file_name,
                                    "page": page_index + 1,
                                    "type": "table"
                                }
                            })

        return final_documents, image_metadata

if __name__ == "__main__":
    processor = PDFProcessor("data/raw/EHF2-10-44.pdf")
    docs, images = processor.process_file()
    
    print(f"✅ Processed {len(docs)} text/table chunks.")
    print(f"✅ Extracted {len(images)} images to disk.")
    
    # Check the first chunk structure
    if docs:
        print(f"Sample Metadata: {docs[0]['metadata']}")