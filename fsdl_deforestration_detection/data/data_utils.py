import os
import subprocess


def decompress_tar_7z(fn: str, input_dir: str, output_dir: str):
    """Decompress the image directories, which are compressed in
    .tar.7z files.

    Args:
        fn (str): Name of the compressed file.
        input_dir (str): Path where the compressed file is in.
        output_dir (str): Path to where we want to decompress
        the file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    subprocess.run(
        [
            "7z",
            "x",
            "-so",
            f"{input_dir}{fn}",
            "|",
            "tar",
            "xf",
            "-",
            "-C",
            output_dir,
        ]
    )
