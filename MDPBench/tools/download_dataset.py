import argparse

def main():
    parser = argparse.ArgumentParser(description="Download MDPBench dataset and ground truth.")
    parser.add_argument("--source", type=str, default="huggingface", choices=["huggingface", "modelscope"], help="Download source")
    parser.add_argument("--repo_id", type=str, default=None, help="Repository ID (e.g., org/repo_name). Defaults to Delores-Lin/MDPBench for HF, DeloresLin/MDPBench for MS")
    parser.add_argument("--local_dir", type=str, default="./MDPBench_dataset", help="Local directory to download the dataset")
    parser.add_argument("--repo_type", type=str, default="dataset", help="Repository type (usually 'dataset')")
    
    args = parser.parse_args()

    if args.repo_id is None:
        if args.source == "modelscope":
            args.repo_id = "DeloresLin/MDPBench"
        else:
            args.repo_id = "Delores-Lin/MDPBench"

    print(f"Downloading dataset from {args.source} repo: {args.repo_id}...")
    
    if args.source == "huggingface":
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir,
            local_dir_use_symlinks=False
        )
    elif args.source == "modelscope":
        from modelscope import snapshot_download
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            local_dir=args.local_dir
        )
        
    print(f"Dataset successfully downloaded to: {args.local_dir}")

if __name__ == "__main__":
    main()
