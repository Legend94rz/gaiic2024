import os
from subprocess import Popen, PIPE
from pathlib import Path


def invoke(input_dir, output_path):
    #Popen("bash test.sh".split())
    input_dir = Path(input_dir)
    data_root = input_dir.parent
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"input_dir: {input_dir}")
    print(f"data_root: {data_root}")
    print(f"output_path: {output_path}")
    # os.system(f"bash test.sh '{input_dir}' '{data_root}' '{output_path}'")
    subproc = Popen(["bash", "test.sh", input_dir, data_root, output_path, input_dir.name], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
    output, err = subproc.communicate()
    subproc.wait()
    print(f"output:\n {output}", flush=True)
    print(f"err:\n {err}", flush=True)


if __name__ == "__main__":
    invoke("/home/renzhen/userdata/repo/gaiic2024/data/track1-A/test", "/home/renzhen/userdata/repo/gaiic2024/tmp/pred.json")
