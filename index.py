import os
from subprocess import Popen
from pathlib import Path


def invoke(input_dir, output_path):
    #Popen("bash test.sh".split())
    data_root = Path(input_dir).parent
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    os.system(f"bash test.sh '{input_dir}' '{data_root}' '{output_path}'")


if __name__ == "__main__":
    invoke("/home/renzhen/userdata/repo/gaiic2024/data/track1-A/test", "/home/renzhen/userdata/repo/gaiic2024/results/pred.json")
