import os
import torch
from pathlib import Path
import json
from utils.manifest import VGManifest
import subprocess
import shutil
import regex
from termcolor import colored
import codecs
from collections import deque

def memo_vg(
    manifests: list[VGManifest],
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    overwrite_output: bool = False,
    seed: int = 42,
    output_resolution: int = 512,
    fps: int = 30,
    num_generated_frames_per_clip: int = 16,
    num_init_past_frames: int = 16,
    num_past_frames: int = 16,
    inference_steps: int = 20,
    cfg_scale: float = 3.5,
    use_xformers: bool = True,
    pipeline_working_directory: Path = Path(__file__).parent / "memo-pipeline",
) -> list[tuple[VGManifest, str]]:

    vg_manifest_file = pipeline_working_directory / "vg-manifest.jsonl"
    vg_manifest = [
        json.dumps(
            dict(
                input_audio_path=str(manifest.audio_path),
                reference_image_path=str(manifest.face_reference_path),
                output_path=str(manifest.output_path),
                target_emotion=manifest.target_emotion
            )
        )
        for manifest in manifests
    ]

    with open(vg_manifest_file, "w+") as f:
        f.write("\n".join(vg_manifest))

    pixi_executable = shutil.which("pixi")
    if pixi_executable is None:
        raise RuntimeError(
            "pixi (https://pixi.prefix.dev) is required to run memo pipeline"
        )

    env = os.environ.copy()
    env["FORCE_COLOR"] = "1"
    env["PY_COLORS"] = "1"
    env["TERM"] = "xterm-256color"
    process = subprocess.Popen(
        [
            pixi_executable,
            "run",
            "python",
            "main.py",
            vg_manifest_file,
            "--device",
            str(device),
            *(["--overwrite"] if overwrite_output else []),
            "--seed",
            str(seed),
            "--weight_dtype",
            str(dtype).split(".")[-1],  # torch.bfloat16 -> bfloat16
            "--output_resolution",
            str(output_resolution),
            "--fps",
            str(fps),
            "--num_generated_frames_per_clip",
            str(num_generated_frames_per_clip),
            "--num_init_past_frames",
            str(num_init_past_frames),
            "--num_past_frames",
            str(num_past_frames),
            "--inference_steps",
            str(inference_steps),
            "--cfg_scale",
            str(cfg_scale),
            *(["--use_xformers"] if use_xformers else []),
        ],
        cwd=pipeline_working_directory,
        text=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env
    )

    decoder = codecs.getincrementaldecoder("utf-8")()
    stdout = deque([""], maxlen=10)
    # carriage_position = 0
    while True:
        char = process.stdout.read(1)
        
        if not char and process.poll() is not None:
            break
        
        decoded_char = None
        try:
            decoded_char = decoder.decode(char)
        except UnicodeDecodeError:
            print("?", end='', flush=True)

        if decoded_char:
            print(decoded_char, end='', flush=True)

            # if decoded_char in ['\r', '\n']:
            #     carriage_position = 0
            if decoded_char == '\n':
                stdout.append("")
                continue

            stdout[-1] += decoded_char
            # stdout[-1] = stdout[-1][:carriage_position] + decoded_char + stdout[-1][carriage_position+1:]
            # carriage_position += 1

    if stdout[-1] == '':
        stdout.pop()

    returncode = process.poll()

    if returncode != 0:
        r = regex.compile(r"(?<=Logs written to ).+\.jsonl")
        match = next(filter(r.match, reversed(stdout)), None)
        if match:
            errored_manifests: list[tuple[VGManifest, str]] = []
            with open(Path(match).relative_to(vg_manifest_file), "r") as f:
                for line in f.readlines():
                    log = json.loads(line)
                    manifest, stacktrace = log["manifest"], log["stacktrace"]
                    manifest_index = vg_manifest.index(manifest)
                    if manifest_index < 0:
                        print(
                            colored(
                                "Could not find original of the errored manifest: "
                                + line,
                                color="red",
                            )
                        )
                        continue
                    errored_manifests.append((manifests[manifest_index], stacktrace))
            return errored_manifests

    return []
