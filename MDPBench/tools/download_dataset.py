import argparse
import base64
import csv
import hashlib
import json
import sys
from pathlib import Path

DEFAULT_TSV_MD5 = {
    "MDPBench_public.tsv": "6aa9a03dcea532be3f92c81635d21883",
}


def _set_csv_field_limit():
    max_size = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_size)
            return
        except OverflowError:
            max_size = int(max_size / 10)


def _find_tsv_file(local_dir, filename):
    local_dir = Path(local_dir)
    direct_path = local_dir / filename
    if direct_path.exists():
        return direct_path

    matches = list(local_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {local_dir}")
    if len(matches) > 1:
        print(f"Found multiple {filename} files, using: {matches[0]}")
    return matches[0]


def _calculate_md5(file_path, chunk_size=1024 * 1024 * 16):
    md5 = hashlib.md5()
    with Path(file_path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5.update(chunk)
    return md5.hexdigest()


def verify_md5(file_path, expected_md5):
    if not expected_md5:
        return None

    actual_md5 = _calculate_md5(file_path)
    if actual_md5.lower() != expected_md5.lower():
        raise ValueError(
            f"MD5 mismatch for {file_path}: expected {expected_md5}, got {actual_md5}"
        )
    print(f"MD5 verified for {file_path}: {actual_md5}")
    return actual_md5


def convert_tsv_dataset(tsv_path, local_dir, image_dir_name, json_filename, overwrite=False, expected_md5=None):
    tsv_path = Path(tsv_path)
    local_dir = Path(local_dir)
    image_dir = local_dir / image_dir_name
    json_path = local_dir / json_filename
    tmp_json_path = json_path.with_suffix(json_path.suffix + ".tmp")

    local_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    _set_csv_field_limit()
    verify_md5(tsv_path, expected_md5)

    count = 0
    with tsv_path.open("r", newline="", encoding="utf-8") as f_in, tmp_json_path.open("w", encoding="utf-8") as f_json:
        reader = csv.DictReader(f_in, delimiter="\t")
        required_fields = {"index", "image", "answer"}
        missing_fields = required_fields - set(reader.fieldnames or [])
        if missing_fields:
            raise ValueError(f"TSV missing required columns: {sorted(missing_fields)}")

        f_json.write("[\n")
        for row in reader:
            image_name = row["index"].strip()
            if not image_name:
                raise ValueError(f"TSV row {count + 2} has an empty index field")

            annotation = json.loads(row["answer"])
            annotation.setdefault("page_info", {})["image_path"] = image_name

            image_path = image_dir / image_name
            if overwrite or not image_path.exists():
                image_bytes = base64.b64decode(row["image"], validate=True)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)

            if count:
                f_json.write(",\n")
            json.dump(annotation, f_json, ensure_ascii=False)
            count += 1

        f_json.write("\n]\n")

    tmp_json_path.replace(json_path)
    print(f"Converted {count} samples from {tsv_path}")
    print(f"Images saved to: {image_dir}")
    print(f"Ground truth saved to: {json_path}")
    return count

def main():
    parser = argparse.ArgumentParser(description="Download MDPBench dataset and ground truth.")
    parser.add_argument("--source", type=str, default="huggingface", choices=["huggingface", "modelscope"], help="Download source")
    parser.add_argument("--repo_id", type=str, default=None, help="Repository ID (e.g., org/repo_name). Defaults to Delores-Lin/MDPBench for HF, DeloresLin/MDPBench_tsv for MS")
    parser.add_argument("--local_dir", type=str, default="./MDPBench_dataset", help="Local directory to download the dataset")
    parser.add_argument("--repo_type", type=str, default="dataset", help="Repository type (usually 'dataset')")
    parser.add_argument("--tsv_path", type=str, default=None, help="Convert a local TSV file instead of downloading it first")
    parser.add_argument("--tsv_filename", type=str, default="MDPBench_public.tsv", help="TSV filename to convert after ModelScope download")
    parser.add_argument("--image_dir_name", type=str, default="MDPBench_img_public", help="Output image directory name")
    parser.add_argument("--json_filename", type=str, default="MDPBench_public.json", help="Output ground-truth JSON filename")
    parser.add_argument("--md5", type=str, default=None, help="Expected MD5 for the TSV file. Defaults to the known checksum for MDPBench_public.tsv")
    parser.add_argument("--skip_md5", action="store_true", help="Skip TSV MD5 verification")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite images that already exist during TSV conversion")
    
    args = parser.parse_args()

    if args.tsv_path:
        expected_md5 = None if args.skip_md5 else args.md5 or DEFAULT_TSV_MD5.get(Path(args.tsv_path).name)
        convert_tsv_dataset(
            tsv_path=args.tsv_path,
            local_dir=args.local_dir,
            image_dir_name=args.image_dir_name,
            json_filename=args.json_filename,
            overwrite=args.overwrite,
            expected_md5=expected_md5,
        )
        return

    if args.repo_id is None:
        if args.source == "modelscope":
            args.repo_id = "DeloresLin/MDPBench_tsv"
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
        tsv_path = _find_tsv_file(args.local_dir, args.tsv_filename)
        expected_md5 = None if args.skip_md5 else args.md5 or DEFAULT_TSV_MD5.get(Path(args.tsv_filename).name)
        convert_tsv_dataset(
            tsv_path=tsv_path,
            local_dir=args.local_dir,
            image_dir_name=args.image_dir_name,
            json_filename=args.json_filename,
            overwrite=args.overwrite,
            expected_md5=expected_md5,
        )
        
    print(f"Dataset successfully downloaded to: {args.local_dir}")

if __name__ == "__main__":
    main()
