import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from itercomp.experiment import merge_runs


def main():
    parser = argparse.ArgumentParser(description="Merge completed disjoint compression or reader shards")
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(merge_runs(args.paths, args.output_dir))


if __name__ == "__main__":
    main()
