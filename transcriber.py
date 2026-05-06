import json
import os
import re
import site
import time

from config import BASE_OUTPUT, METADATA_FILE, MP3_DIR, TXT_DIR, ensure_data_dirs

DLL_DIRECTORY_HANDLES = []


def add_nvidia_dll_directories():
    """Make NVIDIA runtime wheels visible to Windows DLL loading."""
    if os.name != 'nt':
        return

    bin_dirs = []

    for site_packages in site.getsitepackages():
        nvidia_dir = os.path.join(site_packages, "nvidia")
        if not os.path.isdir(nvidia_dir):
            continue

        for package_name in os.listdir(nvidia_dir):
            bin_dir = os.path.join(nvidia_dir, package_name, "bin")
            if os.path.isdir(bin_dir):
                bin_dirs.append(bin_dir)

    if not bin_dirs:
        return

    os.environ["PATH"] = os.pathsep.join(bin_dirs + [os.environ.get("PATH", "")])

    if hasattr(os, "add_dll_directory"):
        for bin_dir in bin_dirs:
            DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(bin_dir))


add_nvidia_dll_directories()

import ctranslate2
from faster_whisper import BatchedInferencePipeline, WhisperModel


ensure_data_dirs()


def sanitize_filename(filename):
    """Convert podcast title to valid filename."""
    filename = re.sub(r'[<>:"/\\|?*]', "", filename)
    filename = re.sub(r"\s+", "_", filename)
    return filename[:200]


def load_episode_metadata():
    """Load episode metadata to get titles."""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def migrate_old_transcript(file, new_txt_path):
    """Rename or remove old hashed transcript files."""
    old_txt_path = os.path.join(TXT_DIR, file + ".txt")
    if not os.path.exists(old_txt_path):
        return False

    if os.path.exists(new_txt_path):
        os.remove(old_txt_path)
        return False

    os.rename(old_txt_path, new_txt_path)
    return True


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name, default):
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def detect_device():
    """Use CUDA when CTranslate2 can see it; otherwise fall back to CPU."""
    forced_device = os.getenv("WHISPER_DEVICE")
    if forced_device:
        return forced_device.strip().lower()

    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def default_model_for(device):
    # large-v3 is the accuracy-first default on GPU. On CPU it is usually too
    # slow for a whole podcast archive, so medium is the practical default.
    return "large-v3" if device == "cuda" else "medium"


def log_message(message, log=None):
    print(message)
    if log:
        log(message)


def cuda_runtime_error(error):
    message = str(error).lower()
    return any(
        part in message
        for part in (
            "cublas",
            "cudnn",
            "cuda",
            "cufft",
            "curand",
            "cusolver",
            "cusparse",
        )
    )


def load_transcription_model(log=None, device_override=None):
    device = device_override or detect_device()
    model_name = os.getenv("WHISPER_MODEL", default_model_for(device))
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE") or (
        "float16" if device == "cuda" else "int8"
    )
    cpu_threads = env_int("WHISPER_CPU_THREADS", max(1, os.cpu_count() or 1))
    num_workers = env_int("WHISPER_NUM_WORKERS", 1)

    load_attempts = [(device, compute_type)]
    if device == "cuda" and compute_type != "int8_float16":
        load_attempts.append(("cuda", "int8_float16"))
    if device == "cuda":
        load_attempts.append(("cpu", "int8"))

    last_error = None
    for attempt_device, attempt_compute_type in load_attempts:
        log_message(
            (
                f"Loading faster-whisper model: {model_name} "
                f"(device={attempt_device}, compute_type={attempt_compute_type})"
            ),
            log,
        )

        try:
            model = WhisperModel(
                model_name,
                device=attempt_device,
                compute_type=attempt_compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
            return BatchedInferencePipeline(model=model), attempt_device
        except Exception as e:
            last_error = e
            log_message(f"Model load failed on {attempt_device}: {str(e)}", log)

    raise RuntimeError(f"Could not load Whisper model: {last_error}")


def transcribe_audio(model, mp3_path, options):
    started = time.perf_counter()
    segments, info = model.transcribe(mp3_path, **options)
    text = "".join(segment.text for segment in segments).strip()
    elapsed = time.perf_counter() - started
    return text, info, elapsed


def get_episode_files():
    files = [f for f in os.listdir(MP3_DIR) if f.lower().endswith(".mp3")]
    metadata = load_episode_metadata()
    metadata_dict = {ep["file"]: ep for ep in metadata} if isinstance(metadata, list) else metadata

    files_with_metadata = []
    for file in files:
        episode_data = metadata_dict.get(file, {})
        episode_num = episode_data.get("episode_number", 999)
        files_with_metadata.append((file, episode_num, episode_data))

    files_with_metadata.sort(key=lambda item: item[1])
    return files_with_metadata


def run_transcriptions(log=None):
    language = os.getenv("WHISPER_LANGUAGE", "it")
    beam_size = env_int("WHISPER_BEAM_SIZE", 5)
    batch_size = env_int("WHISPER_BATCH_SIZE", 16)
    vad_filter = env_bool("WHISPER_VAD_FILTER", True)
    condition_on_previous_text = env_bool("WHISPER_CONDITION_PREVIOUS", True)
    initial_prompt = os.getenv("WHISPER_INITIAL_PROMPT") or None
    hotwords = os.getenv("WHISPER_HOTWORDS") or None
    transcription_options = {
        "language": language,
        "beam_size": beam_size,
        "batch_size": batch_size,
        "temperature": 0.0,
        "condition_on_previous_text": condition_on_previous_text,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "vad_filter": vad_filter,
        "vad_parameters": {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        "hallucination_silence_threshold": 2.0,
    }

    model, device = load_transcription_model(log)
    files = get_episode_files()
    total = len(files)

    if total == 0:
        log_message("No MP3 files found to transcribe", log)
        return

    for i, (file, episode_num, episode_data) in enumerate(files, start=1):
        title = episode_data.get("title", file.replace(".mp3", ""))
        safe_filename = sanitize_filename(title)

        log_message(f"Transcribe {i}/{total}: {title}", log)

        mp3_path = os.path.join(MP3_DIR, file)
        txt_path = os.path.join(TXT_DIR, f"{episode_num}_{safe_filename}.txt")

        if migrate_old_transcript(file, txt_path):
            log_message(
                f"Migrated old transcript to: {episode_num}_{safe_filename}.txt",
                log,
            )

        if os.path.exists(txt_path):
            log_message(f"Already transcribed: {safe_filename}", log)
            continue

        try:
            try:
                text, info, elapsed = transcribe_audio(
                    model,
                    mp3_path,
                    transcription_options,
                )
            except Exception as e:
                if device != "cuda" or not cuda_runtime_error(e):
                    raise

                log_message(
                    (
                        "CUDA transcription failed, falling back to CPU. "
                        "Install CUDA/cuDNN to use GPU acceleration."
                    ),
                    log,
                )
                model, device = load_transcription_model(log, device_override="cpu")
                text, info, elapsed = transcribe_audio(
                    model,
                    mp3_path,
                    transcription_options,
                )

            tmp_path = txt_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp_path, txt_path)

            duration = getattr(info, "duration", 0) or 0
            speed = duration / elapsed if elapsed > 0 else 0
            detected_language = getattr(info, "language", "unknown")
            log_message(
                (
                    f"Transcribed ({detected_language}, {speed:.1f}x realtime, "
                    f"device={device}): {safe_filename}"
                ),
                log,
            )

        except Exception as e:
            log_message(f"Error transcribing {file}: {str(e)}", log)
