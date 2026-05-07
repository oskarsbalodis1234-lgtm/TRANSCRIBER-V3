import os
import threading
import time
import traceback

from flask import Flask, Response, render_template_string, request, send_file

from config import ZIP_PATH, ensure_data_dirs


app = Flask(__name__)

LOG = []
MAX_LOG_LINES = 1000
JOB_LOCK = threading.Lock()
JOB_RUNNING = False


def log(msg):
    print(msg, flush=True)
    LOG.append(msg)
    if len(LOG) > MAX_LOG_LINES:
        del LOG[: len(LOG) - MAX_LOG_LINES]


def set_job_running(value):
    global JOB_RUNNING
    with JOB_LOCK:
        JOB_RUNNING = value


def is_job_running():
    with JOB_LOCK:
        return JOB_RUNNING


def run_pipeline(rss_url):
    LOG.clear()
    set_job_running(True)

    log("state:starting_pipeline")
    log(f"rss:{rss_url}")

    try:
        ensure_data_dirs()

        log("state:fetch_rss")
        from downloader import ingest_rss
        episode_list = ingest_rss(rss_url)

        total = len(episode_list)
        log(f"total_episodes:{total}")

        log("state:downloading")
        from downloader import run_downloads
        run_downloads(episode_list, log)

        log("state:transcribing")
        from transcriber import run_transcriptions
        run_transcriptions(episode_list, log)

        log("state:zipping")
        from main import zip_and_cleanup
        zip_and_cleanup(log)

        log("state:done")

    except Exception as e:
        log("state:error")
        log(str(e))
        log(traceback.format_exc())
    finally:
        set_job_running(False)


@app.route("/")
def home():
    return render_template_string(
        """
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Podcast Transcriber</title>
            <style>
                body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 900px; }
                form { display: flex; gap: .5rem; margin-bottom: 1rem; }
                input { flex: 1; min-width: 0; padding: .55rem .65rem; }
                button, a.button { padding: .55rem .8rem; }
                pre { background: #111; color: #eee; min-height: 360px; padding: 1rem; overflow: auto; white-space: pre-wrap; }
            </style>
        </head>
        <body>
            <h1>Podcast Transcriber</h1>

            <form action="/run">
                <input name="rss" placeholder="Paste RSS link" required>
                <button type="submit">Start</button>
            </form>

            <p>
                <a class="button" href="/download">Download ZIP</a>
                <button onclick="if(confirm('Clear all logs and transcripts?')) fetch('/reset').then(() => location.reload())" style="margin-left: 10px; color: #ff4444; background: none; border: 1px solid #ff4444; border-radius: 4px; cursor: pointer;">Reset System</button>
            </p>
            <pre id="logbox"></pre>

            <script {%- if is_job_running() %} data-running="true" {% endif -%}>
                const box = document.getElementById("logbox");
                const source = new EventSource("/stream");

                source.onmessage = function(event) {
                    if (event.data === "__keepalive__") return;
                    box.textContent += event.data + "\\n";
                    box.scrollTop = box.scrollHeight;
                    
                    if (event.data === "state:done" || event.data.startsWith("state:error")) {
                        source.close();
                    }
                };
            </script>
        </body>
        </html>
        """,
        is_job_running=is_job_running
    )


@app.route("/run")
def run():
    rss = request.args.get("rss", "").strip()

    if not rss:
        return "No RSS provided", 400

    if is_job_running():
        return "A transcription job is already running", 409

    thread = threading.Thread(target=run_pipeline, args=(rss,), daemon=True)
    thread.start()

    return "Running..."


@app.route("/reset")
def reset():
    if is_job_running():
        return "Cannot reset while a job is running", 409

    import shutil
    from config import MP3_DIR, TXT_DIR, METADATA_FILE, BASE_OUTPUT

    try:
        # 1. Clear database content (safer than deleting the file while connected)
        from db import clear_db
        clear_db()

        # 2. Delete data folders
        for folder in [MP3_DIR, TXT_DIR]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        
        # 3. Delete metadata and zip files
        for file_path in [METADATA_FILE, ZIP_PATH]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass

        # 4. Recreate clean directory structure
        ensure_data_dirs()

        # 5. Clear memory logs last
        LOG.clear()
        log("System reset: All files and logs cleared.")
        return "Success", 200
    except Exception as e:
        log(f"Reset failed: {str(e)}")
        return f"Error: {str(e)}", 500


@app.route("/stream")
def stream():
    def generate():
        last = 0

        while True:
            if last > len(LOG): # Handle log reset/clear
                last = 0

            if len(LOG) > last:
                for i in range(last, len(LOG)):
                    yield f"data: {LOG[i]}\n\n"
                last = len(LOG)
            
            # Detect if thread died without setting state:done
            if not is_job_running() and last >= len(LOG):
                if len(LOG) > 0 and "state:done" not in LOG and "state:error" not in LOG:
                    yield "data: Error: The background process crashed (likely Out of Memory).\n\n"
                break

            else:
                yield "data: __keepalive__\n\n"

            time.sleep(1)
            
    response = Response(generate(), mimetype="text/event-stream")
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Disables buffering on Nginx/Koyeb
    response.headers['Connection'] = 'keep-alive'
    return response
    
@app.route("/download")
def download():
    if os.path.exists(ZIP_PATH):
        return send_file(ZIP_PATH, as_attachment=True)

    return "No file yet", 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Server started on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port, threaded=True)
