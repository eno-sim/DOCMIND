from app.ingestion import ingest_file

def main():
    # 1. Define the path to a test file
    test_file_path = "sample.txt" 
    
    # A realistic, multi-paragraph text block to test word boundaries and paragraphs
    sample_text = """Retrieval-Augmented Generation (RAG) is an AI framework for improving the quality of LLM-generated responses by grounding the model on external sources of knowledge. Implementing RAG in an answering system has two main benefits: It ensures that the model has access to the most current, reliable facts, and that users have access to the model's sources, ensuring that its claims can be checked for accuracy.

By grounding an LLM on a set of external, verifiable facts, the model has fewer opportunities to pull information baked into its parameters. This reduces the chances that an LLM will leak sensitive data, or "hallucinate" incorrect or misleading information.

However, to make this process efficient, large documents cannot be fed into the LLM all at once. The context window of a language model is limited, and processing massive amounts of text is computationally expensive. Therefore, documents must be processed through an ingestion pipeline. During ingestion, text is extracted from various formats, normalized to remove noise, and divided into smaller, manageable pieces known as "chunks".

Chunking strategies play a vital role in RAG performance. If chunks are too large, the system risks retrieving irrelevant information, diluting the context. If chunks are too small, they may lack the necessary context to be useful. To mitigate boundary issues where a crucial sentence is split in half, systems typically employ an overlap strategy. This means that the end of one chunk is repeated at the beginning of the next chunk, acting as a linguistic bridge that preserves semantic meaning across artificial boundaries."""
    
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    print(f"Testing ingestion on: {test_file_path}")
    
    try:
        # 2. Call your central router
        # Let's use 200 characters to simulate real chunking constraints
        result = ingest_file(
            path=test_file_path, 
            chunk_size=200,      
            chunk_overlap=40
        )
        
        # 3. Print the results
        print(f"\nModality: {result.modality}")
        print(f"Total Chunks Generated: {len(result.chunks)}")
        
        print("\n--- First 3 Chunks ---")
        for i, chunk in enumerate(result.chunks[:3]):
            print(f"\nChunk {i}: (ID: {chunk.chunk_id})")
            print(f"Text: '{chunk.text}'")
            print(f"Metadata: {chunk.metadata}")
            
        print("\n--- Last Chunk ---")
        last_chunk = result.chunks[-1]
        print(f"\nChunk {len(result.chunks)-1}: (ID: {last_chunk.chunk_id})")
        print(f"Text: '{last_chunk.text}'")
        print(f"Metadata: {last_chunk.metadata}")
            
    except Exception as e:
        print(f"Ingestion failed: {e}")

if __name__ == "__main__":
    main()