import argparse
import os
import shutil

def main():
    parser = argparse.ArgumentParser(description="Copy linear model to production.")
    parser.add_argument("--src_model", required=True, help="Path to source model pkl.")
    parser.add_argument("--prod_model", required=True, help="Path to final (prod) model.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.prod_model), exist_ok=True)

    shutil.copyfile(args.src_model, args.prod_model)
    print(f"[INFO] Model copied from {args.src_model} to {args.prod_model}")

if __name__ == "__main__":
    main()