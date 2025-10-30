import argparse
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix_file", type=str, help="Text file specifying the data mix.")
    parser.add_argument(
        "-d", "--out-dir", default="data", help="Download directory (default: data)"
    )
    parser.add_argument(
        "-x", "--connections", type=int, default=8, help="Max connections per server"
    )
    parser.add_argument("-s", "--segments", type=int, default=8, help="Split segments per file")
    parser.add_argument("-j", "--jobs", type=int, default=4, help="Parallel files")
    args = parser.parse_args()

    tokenizer = "allenai/dolma3-tokenizer"
    data_root = "http://olmo-data.org/"

    tmpfile = tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt")

    with open(args.mix_file) as f:
        for line in f:
            """
            line in the mix file looks like this:
            common_crawl_adult_content,preprocessed/dolma3-0625/v0.1-official/{TOKENIZER}/common_crawl/adult_content/000000.npy
            
            we want to convert it to this:
            http://olmo-data.org/preprocessed/dolma3-0625/v0.1-official/allenai/dolma3-tokenizer/common_crawl/adult_content/000000.npy
                out=preprocessed.dolma3-0625.v0.1-official.allenai.dolma3-tokenizer.common_crawl.adult_content.000000.npy
            """
            line = line.rstrip()
            line = line.split(",")[-1]
            path = line.replace("{TOKENIZER}", tokenizer)
            output_path = ".".join(path.split("/"))
            path = data_root + path
            tmpfile.write(f"{path}\n    out={output_path}\n")

    tmpfile.close()
    print(f"Generated aria2c list: {tmpfile.name}")

    cmd = [
        "aria2c",
        "-i",
        tmpfile.name,
        "-x",
        str(args.connections),
        "-s",
        str(args.segments),
        "-j",
        str(args.jobs),
        "-c",
        "--auto-file-renaming=false",
        "-d",
        args.out_dir,
        "--summary-interval=0",
    ]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
