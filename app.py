from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import subprocess
import sys
import uuid
import json
import shutil
import threading
import re
import os

# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

# Maximum upload size: 250 MB
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024

# ============================================================
# PROJECT FOLDERS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
PROCESSED_FOLDER = BASE_DIR / "processed"
RECORDINGS_FOLDER = BASE_DIR / "recordings"

for folder in (
    UPLOAD_FOLDER,
    PROCESSED_FOLDER,
    RECORDINGS_FOLDER
):
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# ALLOWED AUDIO FILES
# ============================================================

ALLOWED_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac"
}

# ============================================================
# JOB STORAGE
# ============================================================

JOBS = {}
JOBS_LOCK = threading.Lock()

# ============================================================
# WHISPER MODEL
# ============================================================

WHISPER_MODEL = None
WHISPER_LOCK = threading.Lock()

WHISPER_MODEL_SIZE = os.getenv(
    "WHISPER_MODEL_SIZE",
    "base"
)


# ============================================================
# JOB FUNCTIONS
# ============================================================

def set_job(song_id, **values):
    with JOBS_LOCK:
        JOBS.setdefault(song_id, {}).update(values)


def get_job(song_id):
    with JOBS_LOCK:
        return dict(JOBS.get(song_id, {}))


# ============================================================
# METADATA FUNCTIONS
# ============================================================

def metadata_path(song_id):
    return PROCESSED_FOLDER / song_id / "metadata.json"


def load_metadata(song_id):
    path = metadata_path(song_id)

    if not path.exists():
        return {}

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return {}


def save_metadata(song_id, data):

    folder = PROCESSED_FOLDER / song_id

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path(song_id).write_text(
        json.dumps(
            data,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


# ============================================================
# LRC PARSER
# ============================================================

def parse_lrc(text):

    result = []

    offset_ms = 0

    for raw in text.splitlines():

        line = raw.strip()

        line = line.replace(
            "\ufeff",
            ""
        )

        if not line:
            continue

        # LRC offset
        offset_match = re.match(
            r"^\[offset:([+-]?\d+)\]",
            line,
            re.I
        )

        if offset_match:

            offset_ms = int(
                offset_match.group(1)
            )

            continue

        # Timestamp
        stamps = re.findall(
            r"\[(\d{1,3}):(\d{2})(?:[\.:](\d{1,3}))?\]",
            line
        )

        if not stamps:
            continue

        # Remove timestamps
        lyric_text = re.sub(
            r"\[(?:\d{1,3}:\d{2}(?:[\.:]\d{1,3})?)\]",
            "",
            line
        ).strip()

        if not lyric_text:
            continue

        for minutes, seconds, fraction in stamps:

            fraction = fraction or "0"

            if len(fraction) == 1:

                milliseconds = (
                    int(fraction) * 100
                )

            elif len(fraction) == 2:

                milliseconds = (
                    int(fraction) * 10
                )

            else:

                milliseconds = int(
                    fraction[:3]
                )

            total = (
                int(minutes) * 60
                + int(seconds)
                + milliseconds / 1000
                + offset_ms / 1000
            )

            if total >= 0:

                result.append({
                    "time": round(total, 3),
                    "start": round(total, 3),
                    "text": lyric_text
                })

    result.sort(
        key=lambda x: x["time"]
    )

    # Remove duplicates
    unique = []
    seen = set()

    for item in result:

        key = (
            item["time"],
            item["text"]
        )

        if key not in seen:

            seen.add(key)

            unique.append(item)

    # Calculate end time
    for i, item in enumerate(unique):

        if i + 1 < len(unique):

            item["end"] = (
                unique[i + 1]["time"]
            )

        else:

            item["end"] = (
                item["time"] + 8
            )

    return unique


# ============================================================
# WRITE LRC
# ============================================================

def write_lrc(song_id, lines):

    path = (
        PROCESSED_FOLDER
        / song_id
        / "lyrics.lrc"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    def stamp(seconds):

        minutes = int(
            seconds // 60
        )

        remaining = (
            seconds
            - minutes * 60
        )

        return (
            f"[{minutes:02d}:"
            f"{remaining:05.2f}]"
        )

    text = "\n".join(
        f"{stamp(float(x['time']))}"
        f"{str(x['text']).strip()}"
        for x in lines
        if str(x.get("text", "")).strip()
    )

    path.write_text(
        text + ("\n" if text else ""),
        encoding="utf-8"
    )

    return path


# ============================================================
# READ LYRICS
# ============================================================

def lyrics_for_song(song_id):

    path = (
        PROCESSED_FOLDER
        / song_id
        / "lyrics.lrc"
    )

    if not path.exists():
        return []

    try:

        return parse_lrc(
            path.read_text(
                encoding="utf-8-sig",
                errors="replace"
            )
        )

    except Exception:

        return []


# ============================================================
# LOAD FASTER WHISPER
# ============================================================

def get_whisper_model():

    global WHISPER_MODEL

    with WHISPER_LOCK:

        if WHISPER_MODEL is None:

            from faster_whisper import WhisperModel

            print()
            print(
                "Loading Faster-Whisper model:"
                f" {WHISPER_MODEL_SIZE}"
            )

            WHISPER_MODEL = WhisperModel(
                WHISPER_MODEL_SIZE,
                device="cpu",
                compute_type="int8"
            )

            print(
                "Faster-Whisper ready."
            )

    return WHISPER_MODEL


# ============================================================
# TRANSCRIBE VOCALS
# ============================================================

def transcribe_vocals(
    song_id,
    vocals_file
):

    model = get_whisper_model()

    segments, info = model.transcribe(
        str(vocals_file),
        beam_size=3,
        vad_filter=True,
        word_timestamps=False,
        condition_on_previous_text=False
    )

    lines = []

    for segment in segments:

        text = (
            segment.text or ""
        ).strip()

        if not text:
            continue

        lines.append({

            "time": round(
                float(segment.start),
                3
            ),

            "start": round(
                float(segment.start),
                3
            ),

            "end": round(
                float(segment.end),
                3
            ),

            "text": text
        })

    write_lrc(
        song_id,
        lines
    )

    language = getattr(
        info,
        "language",
        None
    )

    probability = getattr(
        info,
        "language_probability",
        None
    )

    return (
        lines,
        language,
        probability
    )


# ============================================================
# FIND UPLOADED AUDIO
# ============================================================

def find_audio(song_id):

    folder = (
        UPLOAD_FOLDER
        / song_id
    )

    if not folder.exists():
        return None

    for file in folder.iterdir():

        if (
            file.is_file()
            and file.suffix.lower()
            in ALLOWED_EXTENSIONS
        ):

            return file

    return None


# ============================================================
# FIND DEMUCS OUTPUTS
# ============================================================

def find_demucs_outputs(folder):

    vocals = None
    instrumental = None

    for file in folder.rglob("*"):

        if not file.is_file():
            continue

        name = file.name.lower()

        if name == "vocals.mp3":

            vocals = file

        elif name == "no_vocals.mp3":

            instrumental = file

    return (
        vocals,
        instrumental
    )


# ============================================================
# AI KARAOKE PIPELINE
# ============================================================

def run_ai_pipeline(song_id):

    input_file = find_audio(
        song_id
    )

    song = load_metadata(
        song_id
    )

    output_folder = (
        PROCESSED_FOLDER
        / song_id
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    try:

        # ----------------------------------------
        # Validate uploaded audio
        # ----------------------------------------

        if input_file is None:

            raise RuntimeError(
                "Uploaded audio file was not found."
            )

        # ----------------------------------------
        # Initial status
        # ----------------------------------------

        song["status"] = "processing"
        song["progress"] = 5

        save_metadata(
            song_id,
            song
        )

        set_job(
            song_id,
            status="running",
            progress=5,
            message="Starting AI vocal remover...",
            song=song
        )

        # ----------------------------------------
        # DEMUCS COMMAND
        # ----------------------------------------

        command = [

            sys.executable,

            "-m",
            "demucs",

            "--two-stems=vocals",

            "-n",
            "htdemucs",

            "--mp3",

            "--mp3-bitrate",
            "320",

            "-o",
            str(output_folder),

            str(input_file)
        ]

        print()
        print("=" * 72)
        print(
            "AI VOCAL REMOVER STARTED"
        )
        print("=" * 72)

        print(
            " ".join(command)
        )

        print("=" * 72)

        set_job(
            song_id,
            progress=15,
            message=(
                "AI is removing "
                "the original singer voice..."
            )
        )

        # ----------------------------------------
        # RUN DEMUCS
        # ----------------------------------------

        result = subprocess.run(

            command,

            capture_output=True,

            text=True,

            timeout=3600
        )

        if result.stdout:

            print(
                result.stdout
            )

        if result.stderr:

            print(
                result.stderr
            )

        if result.returncode != 0:

            error_text = (
                result.stderr
                or result.stdout
                or "Demucs failed."
            )

            raise RuntimeError(
                error_text[-5000:]
            )

        # ----------------------------------------
        # FIND OUTPUTS
        # ----------------------------------------

        vocals_file, instrumental_file = (
            find_demucs_outputs(
                output_folder
            )
        )

        if instrumental_file is None:

            raise RuntimeError(
                "AI completed, but the "
                "instrumental track was not found."
            )

        # ----------------------------------------
        # COPY FINAL FILES
        # ----------------------------------------

        final_instrumental = (
            output_folder
            / "instrumental.mp3"
        )

        final_vocals = (
            output_folder
            / "vocals.mp3"
        )

        if (
            instrumental_file.resolve()
            != final_instrumental.resolve()
        ):

            shutil.copy2(
                instrumental_file,
                final_instrumental
            )

        if vocals_file:

            if (
                vocals_file.resolve()
                != final_vocals.resolve()
            ):

                shutil.copy2(
                    vocals_file,
                    final_vocals
                )

        # ----------------------------------------
        # KARAOKE TRACK READY
        # ----------------------------------------

        song["instrumental_url"] = (
            f"/processed/"
            f"{song_id}/"
            f"instrumental.mp3"
        )

        if final_vocals.exists():

            song["vocals_url"] = (
                f"/processed/"
                f"{song_id}/"
                f"vocals.mp3"
            )

        else:

            song["vocals_url"] = None

        song["progress"] = 72

        song["status"] = (
            "lyrics_processing"
        )

        save_metadata(
            song_id,
            song
        )

        set_job(
            song_id,
            progress=72,
            message=(
                "Instrumental ready. "
                "Creating synchronized lyrics..."
            ),
            song=song
        )

        # ----------------------------------------
        # AUTOMATIC LYRICS
        # ----------------------------------------

        lyrics_error = None
        lines = []

        if final_vocals.exists():

            try:

                (
                    lines,
                    language,
                    probability
                ) = transcribe_vocals(
                    song_id,
                    final_vocals
                )

                song["detected_language"] = (
                    language
                )

                song["language_probability"] = (
                    probability
                )

                song["lyrics_available"] = (
                    bool(lines)
                )

                if lines:

                    song["lyrics_source"] = (
                        "faster-whisper"
                    )

                else:

                    song["lyrics_source"] = None

            except Exception as exc:

                lyrics_error = str(exc)

                print(
                    "AUTO LYRICS ERROR:",
                    exc
                )

                song[
                    "lyrics_available"
                ] = False

        else:

            lyrics_error = (
                "Vocal stem was not created."
            )

            song[
                "lyrics_available"
            ] = False

        # ----------------------------------------
        # FINAL READY STATUS
        # ----------------------------------------

        song["status"] = "ready"

        song["progress"] = 100

        song["lyrics_url"] = (
            f"/api/lyrics/{song_id}"
        )

        save_metadata(
            song_id,
            song
        )

        message = (
            "Karaoke ready: "
            "original singer removed."
        )

        if lines:

            message += (
                " Automatic synchronized "
                "lyrics are ready."
            )

        elif lyrics_error:

            message += (
                " Automatic lyrics were "
                "unavailable; LRC upload "
                "is supported."
            )

        set_job(

            song_id,

            status="done",

            progress=100,

            message=message,

            song=song,

            lyrics_error=lyrics_error
        )

        print()
        print(
            "KARAOKE READY:",
            song_id
        )
        print()

    except subprocess.TimeoutExpired:

        song["status"] = "error"
        song["progress"] = 0

        save_metadata(
            song_id,
            song
        )

        set_job(

            song_id,

            status="error",

            progress=0,

            message=(
                "AI processing timed out."
            ),

            song=song
        )

    except Exception as exc:

        print()
        print(
            "AI PIPELINE ERROR:",
            exc
        )

        song["status"] = "error"
        song["progress"] = 0

        save_metadata(
            song_id,
            song
        )

        set_job(

            song_id,

            status="error",

            progress=0,

            message=str(exc),

            song=song
        )


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD SONG
# ============================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_song():

    try:

        audio_file = (
            request.files.get("file")
            or request.files.get("songFile")
            or request.files.get("audio")
        )

        if (
            audio_file is None
            or not audio_file.filename
        ):

            return jsonify(
                success=False,
                error="No audio file received."
            ), 400

        filename = secure_filename(
            audio_file.filename
        )

        extension = (
            Path(filename)
            .suffix
            .lower()
        )

        if extension not in ALLOWED_EXTENSIONS:

            return jsonify(
                success=False,
                error=(
                    "Use MP3, WAV, M4A, "
                    "FLAC, OGG or AAC."
                )
            ), 400

        song_id = uuid.uuid4().hex

        folder = (
            UPLOAD_FOLDER
            / song_id
        )

        folder.mkdir(
            parents=True,
            exist_ok=True
        )

        input_file = (
            folder
            / filename
        )

        audio_file.save(
            str(input_file)
        )

        if (
            not input_file.exists()
            or input_file.stat().st_size == 0
        ):

            return jsonify(
                success=False,
                error="Uploaded file is empty."
            ), 400

        # ----------------------------------------
        # Song metadata
        # ----------------------------------------

        song = {

            "id": song_id,

            "title": Path(
                filename
            ).stem,

            "original_filename": filename,

            "size": input_file.stat().st_size,

            "status": "uploaded",

            "progress": 0,

            "original_url": (
                f"/uploads/"
                f"{song_id}/"
                f"{filename}"
            ),

            "instrumental_url": None,

            "vocals_url": None,

            "lyrics_url": (
                f"/api/lyrics/"
                f"{song_id}"
            ),

            "lyrics_available": False,

            "lyrics_source": None,

            "detected_language": None
        }

        save_metadata(
            song_id,
            song
        )

        # ----------------------------------------
        # AUTOMATIC AI PROCESSING
        # ----------------------------------------

        set_job(

            song_id,

            status="queued",

            progress=1,

            message=(
                "Song uploaded. "
                "AI vocal remover "
                "is starting..."
            ),

            song=song
        )

        threading.Thread(

            target=run_ai_pipeline,

            args=(song_id,),

            daemon=True

        ).start()

        return jsonify(

            success=True,

            message=(
                "Song uploaded. "
                "AI vocal remover "
                "started automatically."
            ),

            song=song
        )

    except Exception as exc:

        print(
            "UPLOAD ERROR:",
            exc
        )

        return jsonify(

            success=False,

            error=str(exc)

        ), 500


# ============================================================
# MANUAL SEPARATION
# ============================================================

@app.route(
    "/api/separate/<song_id>",
    methods=["POST"]
)
def start_separation(song_id):

    song = load_metadata(
        song_id
    )

    if not song:

        return jsonify(
            success=False,
            error="Song not found."
        ), 404

    job = get_job(
        song_id
    )

    if job.get("status") in {
        "queued",
        "running"
    }:

        return jsonify(

            success=True,

            status=job.get(
                "status"
            ),

            progress=job.get(
                "progress",
                0
            ),

            message=job.get(
                "message"
            ),

            song=song
        )

    set_job(

        song_id,

        status="queued",

        progress=1,

        message=(
            "AI vocal remover queued..."
        ),

        song=song
    )

    threading.Thread(

        target=run_ai_pipeline,

        args=(song_id,),

        daemon=True

    ).start()

    return jsonify(

        success=True,

        status="queued",

        progress=1,

        message=(
            "AI vocal remover started."
        ),

        song=song
    )


# ============================================================
# SEPARATION STATUS
# ============================================================

@app.route(
    "/api/separate/status/<song_id>"
)
def separation_status(song_id):

    job = get_job(
        song_id
    )

    song = load_metadata(
        song_id
    )

    if not song:

        return jsonify(

            success=False,

            error="Song not found."

        ), 404

    return jsonify(

        success=True,

        status=job.get(

            "status",

            song.get(
                "status",
                "uploaded"
            )
        ),

        progress=job.get(

            "progress",

            song.get(
                "progress",
                0
            )
        ),

        message=job.get(

            "message",

            "Waiting..."
        ),

        lyrics_error=job.get(
            "lyrics_error"
        ),

        song=job.get(
            "song",
            song
        )
    )


# ============================================================
# GET LYRICS
# ============================================================

@app.route(
    "/api/lyrics/<song_id>",
    methods=["GET"]
)
def get_lyrics(song_id):

    song = load_metadata(
        song_id
    )

    if not song:

        return jsonify(

            success=False,

            lyrics=[],

            error="Song not found."

        ), 404

    lyrics = lyrics_for_song(
        song_id
    )

    return jsonify(

        success=True,

        available=bool(
            lyrics
        ),

        synced=bool(
            lyrics
        ),

        source=song.get(
            "lyrics_source"
        ),

        language=song.get(
            "detected_language"
        ),

        lyrics=lyrics,

        lyrics_url=(
            f"/processed/"
            f"{song_id}/lyrics.lrc"
            if lyrics
            else None
        )
    )


# ============================================================
# UPLOAD LRC LYRICS
# ============================================================

@app.route(
    "/api/lyrics/<song_id>",
    methods=["POST"]
)
def upload_lrc(song_id):

    song = load_metadata(
        song_id
    )

    if not song:

        return jsonify(

            success=False,

            error="Song not found."

        ), 404

    file = (
        request.files.get("file")
        or request.files.get("lyrics")
        or request.files.get("lrc")
    )

    if (
        file is None
        or not file.filename
    ):

        return jsonify(

            success=False,

            error=(
                "Choose an LRC "
                "or TXT lyrics file."
            )

        ), 400

    raw = file.read(
        5 * 1024 * 1024 + 1
    )

    if len(raw) > 5 * 1024 * 1024:

        return jsonify(

            success=False,

            error="Lyrics file is too large."

        ), 400

    text = raw.decode(
        "utf-8-sig",
        errors="replace"
    )

    lines = parse_lrc(
        text
    )

    if not lines:

        return jsonify(

            success=False,

            error=(
                "No timed lyrics found. "
                "Example: "
                "[00:12.50]Hello"
            )

        ), 400

    write_lrc(
        song_id,
        lines
    )

    song["lyrics_available"] = True

    song["lyrics_source"] = (
        "user-lrc"
    )

    save_metadata(
        song_id,
        song
    )

    return jsonify(

        success=True,

        lyrics=lines,

        message=(
            "Lyrics uploaded successfully."
        )
    )


# ============================================================
# SONG LIBRARY
# ============================================================

@app.route(
    "/api/songs"
)
def songs():

    items = []

    if not PROCESSED_FOLDER.exists():

        return jsonify(
            success=True,
            songs=[]
        )

    for folder in PROCESSED_FOLDER.iterdir():

        if not folder.is_dir():
            continue

        meta = (
            folder
            / "metadata.json"
        )

        if not meta.exists():
            continue

        try:

            song = json.loads(
                meta.read_text(
                    encoding="utf-8"
                )
            )

            song_id = song.get(
                "id",
                folder.name
            )

            # Repair status if instrumental exists
            instrumental = (
                folder
                / "instrumental.mp3"
            )

            if instrumental.exists():

                song["instrumental_url"] = (
                    f"/processed/"
                    f"{song_id}/"
                    f"instrumental.mp3"
                )

                if song.get(
                    "status"
                ) != "error":

                    song["status"] = "ready"

            song["lyrics_available"] = bool(
                lyrics_for_song(
                    song_id
                )
            )

            items.append(
                song
            )

        except Exception as exc:

            print(
                "LIBRARY ERROR:",
                exc
            )

    items.sort(

        key=lambda x: x.get(
            "id",
            ""
        ),

        reverse=True
    )

    return jsonify(

        success=True,

        songs=items
    )


# ============================================================
# ORIGINAL UPLOADED AUDIO
# ============================================================

@app.route(
    "/uploads/<path:filename>"
)
def uploaded_file(filename):

    return send_from_directory(

        UPLOAD_FOLDER,

        filename
    )


# ============================================================
# PROCESSED AUDIO
# ============================================================

@app.route(
    "/processed/<path:filename>"
)
def processed_file(filename):

    return send_from_directory(

        PROCESSED_FOLDER,

        filename
    )


# ============================================================
# RECORDING UPLOAD
# ============================================================

@app.route(
    "/api/recordings",
    methods=["POST"]
)
def upload_recording():

    try:

        file = request.files.get(
            "file"
        )

        if (
            file is None
            or not file.filename
        ):

            return jsonify(

                success=False,

                error="No recording received."

            ), 400

        safe_name = secure_filename(
            file.filename
        )

        extension = (
            Path(safe_name)
            .suffix
            .lower()
        )

        allowed_recording_extensions = {

            ".webm",
            ".ogg",
            ".wav",
            ".mp3",
            ".m4a"
        }

        if extension not in (
            allowed_recording_extensions
        ):

            extension = ".webm"

        filename = (
            uuid.uuid4().hex
            + extension
        )

        path = (
            RECORDINGS_FOLDER
            / filename
        )

        file.save(
            str(path)
        )

        return jsonify(

            success=True,

            recording={

                "filename": filename,

                "url": (
                    f"/recordings/"
                    f"{filename}"
                )
            }
        )

    except Exception as exc:

        print(
            "RECORDING ERROR:",
            exc
        )

        return jsonify(

            success=False,

            error=str(exc)

        ), 500


# ============================================================
# SERVE RECORDINGS
# ============================================================

@app.route(
    "/recordings/<path:filename>"
)
def recording_file(filename):

    return send_from_directory(

        RECORDINGS_FOLDER,

        filename
    )


# ============================================================
# FILE TOO LARGE
# ============================================================

@app.errorhandler(413)
def too_large(_):

    return jsonify(

        success=False,

        error=(
            "File too large. "
            "Maximum is 250 MB."
        )

    ), 413


# ============================================================
# GENERAL ERROR HANDLER
# ============================================================

@app.errorhandler(500)
def server_error(error):

    print(
        "SERVER ERROR:",
        error
    )

    return jsonify(

        success=False,

        error="Internal server error."

    ), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 72)
    print(
        "🎤 AI KARAOKE STUDIO"
    )
    print(
        "🤖 AUTOMATIC AI VOCAL REMOVER"
    )
    print(
        "🎵 AUTOMATIC INSTRUMENTAL GENERATION"
    )
    print(
        "📝 AUTOMATIC SYNCHRONIZED LYRICS"
    )
    print(
        "🎙️ KARAOKE RECORDING"
    )
    print("=" * 72)
    print()
    print(
        "🌐 Open in browser:"
    )
    print(
        "http://127.0.0.1:5000"
    )
    print()
    print("=" * 72)

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=False,

        use_reloader=False
    )