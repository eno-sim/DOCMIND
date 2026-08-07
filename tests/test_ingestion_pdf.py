from app.ingestion import ingest_file

def main():
    # 1. Define the path to a test file
    test_file_path = "/home/n-esimo/DocMind/saint_exupery_il_piccolo_principe.pdf" 
    

        
    print(f"Testing ingestion on: {test_file_path}")
    
    try:
        # 2. Call your central router
        # Let's use 200 characters to simulate real chunking constraints
        result = ingest_file(
            path=test_file_path, 
            chunk_size=2000,      
            chunk_overlap=70
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