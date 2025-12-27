import os
from datasets import load_dataset
from tqdm import tqdm

def download_stream(dataset_id, subset, split, text_column, output_path, max_size_mb=100):
    """
    Stream download a dataset and save to text file until max_size_mb is reached.
    """
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping download.")
        return

    print(f"Starting stream for {dataset_id}...")
    try:
        # Load dataset in streaming mode
        ds = load_dataset(dataset_id, subset, split=split, streaming=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            total_bytes = 0
            max_bytes = max_size_mb * 1024 * 1024
            pbar = tqdm(total=max_size_mb, unit='MB', desc=f"Downloading {dataset_id}")

            for i, sample in enumerate(ds):
                text = sample.get(text_column, "")
                if not text:
                    continue
                
                # Simple cleaning if necessary
                text = text.strip() + "\n<|endoftext|>\n"
                
                # Write to file
                f.write(text)
                
                # Update progress
                bytes_written = len(text.encode('utf-8'))
                total_bytes += bytes_written
                pbar.update(bytes_written / (1024 * 1024))

                if total_bytes >= max_bytes:
                    print(f"\nReached {max_size_mb}MB limit. Stopping.")
                    break
            
            pbar.close()
            print(f"Saved {output_path} ({total_bytes / (1024 * 1024):.2f} MB)")

    except Exception as e:
        print(f"Error downloading {dataset_id}: {e}")

if __name__ == "__main__":
    # Define target datasets
    # 1. English: FineWeb-Edu (High quality educational content)
    # Using 'sample-10BT' subset if available or default
    en_config = {
        "id": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT", 
        "split": "train",
        "text_column": "text",
        "output": "dataset_local_en.txt",
        "size_mb": 100 
    }

    # 2. Chinese: SkyPile-150B (High quality filtered Chinese web data)
    zh_config = {
        "id": "Skywork/SkyPile-150B",
        "subset": None, # 2023-50 or None
        "split": "train",
        "text_column": "text", 
        "output": "dataset_local_zh.txt",
        "size_mb": 100
    }

    print(">>> Starting Local Data Preparation Script <<<")
    print("This script uses HuggingFace Streaming API to fetch small high-quality samples.")
    
    # Download English
    download_stream(
        en_config["id"], 
        en_config["subset"], 
        en_config["split"], 
        en_config["text_column"], 
        en_config["output"],
        en_config["size_mb"]
    )

    # Download Chinese
    download_stream(
        zh_config["id"], 
        zh_config["subset"], 
        zh_config["split"], 
        zh_config["text_column"], 
        zh_config["output"],
        zh_config["size_mb"]
    )

    # Mix Datasets
    mix_output = "dataset_local_mix.txt"
    print(f"\nMixing datasets into {mix_output}...")
    try:
        with open(mix_output, 'w', encoding='utf-8') as outfile:
            for fname in [en_config["output"], zh_config["output"]]:
                if os.path.exists(fname):
                    with open(fname, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            outfile.write(line)
        print(f"Created mixed dataset: {mix_output}")
    except Exception as e:
        print(f"Error mixing datasets: {e}")

    print("\n>>> All Done! <<<")
    print("You can now run training with:")
    print(f"cargo run --release -- train-local --chinese-path {zh_config['output']} --english-path {en_config['output']}")
