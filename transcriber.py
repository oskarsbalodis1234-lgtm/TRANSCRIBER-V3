import json
import os
import re
import random
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

    import ctranslate2
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


def default_model_for(device):
    # large-v3-turbo is the best speed/accuracy balance on GPU. On CPU it is usually too
    # slow for a whole podcast archive, so medium is the practical default.
    return "large-v3-turbo" if device == "cuda" else "tiny.en"


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
    # Lazy import to save RAM on cloud environments
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    device = device_override or detect_device()
    model_name = os.getenv("WHISPER_MODEL", default_model_for(device))
    compute_type = os.getenv("WHISPER_COMPUTE_TYPE") or (
        "float16" if device == "cuda" else "int8"
    )
    cpu_threads = env_int("WHISPER_CPU_THREADS", 1) # Reduce threads to save memory overhead
    num_workers = env_int("WHISPER_NUM_WORKERS", 1)
    use_flash_attention = env_bool("WHISPER_FLASH_ATTENTION", device == "cuda")

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
                flash_attention=use_flash_attention,
            )
            return BatchedInferencePipeline(model=model), attempt_device
        except Exception as e:
            last_error = e
            log_message(f"Model load failed on {attempt_device}: {str(e)}", log)

    raise RuntimeError(f"Could not load Whisper model: {last_error}")


def transcribe_audio(model, mp3_path, options, log=None):
    import requests
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    last_error = "No API keys provided"

    # 1. Try Deepgram SDK (Nova-3)
    if deepgram_key:
        try:
            from deepgram import DeepgramClient
            
            log_message(f"[{os.path.basename(mp3_path)}] Attempting Deepgram API...", log)
            started = time.perf_counter()
            deepgram = DeepgramClient(api_key=deepgram_key)

            with open(mp3_path, "rb") as audio:
                source = {"buffer": audio}
                dg_options = {
                    "model": "nova-3",
                    "smart_format": True,
                    "language": options.get("language", "it")
                }
                response = deepgram.listen.v1.media.transcribe_file(source, dg_options)

            text = response.results.channels[0].alternatives[0].transcript
            duration = response.metadata.duration
            elapsed = time.perf_counter() - started
            
            info = type('obj', (object,), {'duration': duration, 'language': options.get("language") or "it"})
            return text, info, elapsed
        except Exception as e:
            last_error = f"Deepgram failed: {e}"
            log_message(last_error, log)

    # 2. Try Groq (Fast & Free, but strict hourly limits)
    if groq_key:
        file_size = os.path.getsize(mp3_path)
        if file_size > 25 * 1024 * 1024:
            log_message(f"Warning: {os.path.basename(mp3_path)} is {file_size/(1024*1024):.1f}MB. Groq limit is 25MB.", log)

        started = time.perf_counter()
        url = "https://api.groq.com/openai/v1/audio/transcriptions"
        headers = {"Authorization": f"Bearer {groq_key}"}

        try:
            with open(mp3_path, "rb") as f:
                files = {
                    "file": (os.path.basename(mp3_path), f, "audio/mpeg"),
                    "model": (None, "whisper-large-v3"),
                    "language": (None, options.get("language", "it")),
                    "response_format": (None, "verbose_json"),
                }
                log_message(f"[{os.path.basename(mp3_path)}] Attempting Groq API...", log)
                response = requests.post(url, headers=headers, files=files, timeout=300)
                
                if response.status_code == 413:
                    log_message(f"Error: {os.path.basename(mp3_path)} is too large for Groq (Max 25MB).", log)
                else:
                    response.raise_for_status()
                    
                    result = response.json()
                    text = result.get("text", "")
                    duration = result.get("duration", 0)
                    detected_lang = result.get("language") or options.get("language") or "it"
                    elapsed = time.perf_counter() - started
                    
                    # Create a mock info object that looks like the faster-whisper output
                    info = type('obj', (object,), {'duration': duration, 'language': detected_lang})
                    return text, info, elapsed

        except Exception as e:
            last_error = f"Groq failed: {e}"
            log_message(last_error, log)

    # 3. Try OpenAI (Very accurate, paid but reliable)
    if openai_key:
        try:
            file_size = os.path.getsize(mp3_path)
            if file_size > 25 * 1024 * 1024:
                log_message(f"OpenAI error: {os.path.basename(mp3_path)} exceeds 25MB limit.", log)
            else:
                log_message(f"[{os.path.basename(mp3_path)}] Attempting OpenAI API...", log)
                started = time.perf_counter()
                url = "https://api.openai.com/v1/audio/transcriptions"
                headers = {"Authorization": f"Bearer {openai_key}"}
                with open(mp3_path, "rb") as f:
                    files = {
                        "file": (os.path.basename(mp3_path), f, "audio/mpeg"),
                        "model": (None, "whisper-1"),
                        "language": (None, options.get("language", "it")),
                    }
                    response = requests.post(url, headers=headers, files=files, timeout=300)
                    response.raise_for_status()
                
                result = response.json()
                text = result.get("text", "")
                # OpenAI doesn't return duration in the basic response, 
                # we can estimate it if needed, but 0 is safe.
                duration = 0 
                elapsed = time.perf_counter() - started
                info = type('obj', (object,), {'duration': duration, 'language': options.get("language") or "it"})
                return text, info, elapsed
        except Exception as e:
            last_error = f"OpenAI failed: {e}"
            log_message(last_error, log)

    # This point is only reached if all APIs failed or were skipped
    if model is None:
        raise RuntimeError(f"All transcription APIs failed: {last_error}")

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


def run_transcriptions(episode_list=None, log=None):
    language = os.getenv("WHISPER_LANGUAGE", "it")
    beam_size = env_int("WHISPER_BEAM_SIZE", 1)
    batch_size = env_int("WHISPER_BATCH_SIZE", 16)
    vad_filter = env_bool("WHISPER_VAD_FILTER", True)
    
    transcription_options = {
        "language": language,
        "beam_size": beam_size,
        "batch_size": batch_size,
        "temperature": 0.0,
        "condition_on_previous_text": env_bool("WHISPER_CONDITION_PREVIOUS", True),
        "initial_prompt": os.getenv("WHISPER_INITIAL_PROMPT"),
        "hotwords": os.getenv("WHISPER_HOTWORDS"),
        "vad_filter": vad_filter,
        "vad_parameters": {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        "hallucination_silence_threshold": 2.0,
    }

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "").strip()
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()

    model = None
    if groq_key or deepgram_key or openai_key:
        device = "API Cluster"
        active = [p for p, k in [("Deepgram", deepgram_key), ("Groq", groq_key), ("OpenAI", openai_key)] if k]
        log_message(f"Active APIs: {', '.join(active)}. Using remote transcription.", log)
    else:
        log_message("No API keys found. Falling back to local model.", log)
        model, device = load_transcription_model(log)

    # Use provided list or fall back to directory scanning
    if episode_list:
        files_to_process = [(ep["file"], ep.get("episode_number", i), ep) 
                           for i, ep in enumerate(episode_list)]
    else:
        files_to_process = get_episode_files()
        
    total = len(files_to_process)

    if total == 0:
        log_message("No MP3 files found to transcribe", log)
        return

    for i, (file, episode_num, episode_data) in enumerate(files_to_process, start=1):
        title = episode_data.get("title", file.replace(".mp3", ""))
        safe_filename = sanitize_filename(title)

        log_message(f"[{i}/{total}] Processing: {title}", log)

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

        success = False
        attempts = 0
        while not success and attempts < 3:
            try:
                text, info, elapsed = transcribe_audio(
                    model,
                    mp3_path,
                    transcription_options,
                    log
                )
                
                # Pacing: API providers typically require delays to avoid rate limits.
                if device == "API Cluster" and text is not None:
                    pacing_delay = 40 
                    remaining_wait = max(0, pacing_delay - elapsed)
                    if remaining_wait > 0:
                        log_message(f"Pacing: waiting {remaining_wait:.1f}s before next file...", log)
                        time.sleep(remaining_wait)
                
                success = True

            except Exception as e:
                error_msg = str(e)
                # If an API limit was hit, add a cooldown before the next episode
                if "429" in error_msg or "limit" in error_msg.lower():
                    attempts += 1
                    wait = 20 * attempts
                    log_message(f"Rate limit hit. Waiting {wait}s before retrying episode (Attempt {attempts}/3)...", log)
                    time.sleep(wait)
                    continue

                if device != "cuda" or not cuda_runtime_error(e):
                    log_message(f"Error transcribing {file}: {error_msg}", log)
                    break

                log_message(
                    (
                        "CUDA transcription failed, falling back to CPU. "
                        "Install CUDA/cuDNN to use GPU acceleration."
                    ),
                    log,
                )
                model, device = load_transcription_model(log, device_override="cpu")
                # Let the while loop retry with the new CPU model
                continue

            # If transcribe_audio returned None (e.g. skipped due to size), continue to next file
            if text is None:
                break

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
