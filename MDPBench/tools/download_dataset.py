import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download MDPBench dataset and ground truth from Hugging Face.")
    parser.add_argument("--repo_id", type=str, default="Delores-Lin/MDPBench", help="Hugging Face repository ID (e.g., org/repo_name)")
    parser.add_argument("--local_dir", type=str, default="../MDPBench_dataset", help="Local directory to download the dataset")
    parser.add_argument("--repo_type", type=str, default="dataset", help="Repository type (usually 'dataset')")
    
    args = parser.parse_args()

    print(f"Downloading dataset from Hugging Face repo: {args.repo_id}...")
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Dataset successfully downloaded to: {args.local_dir}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Note: If you haven't uploaded the dataset yet or the repo is private, make sure your repo_id is correct and you are logged in (huggingface-cli login).")

if __name__ == "__main__":
    main()
