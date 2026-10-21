import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="Shorts Captioner AI", page_icon="🎬", layout="wide")
st.title("🎬 Shorts Captioner AI")
st.caption("Automatic English transcription · editable captions · animated karaoke-style highlighting · MP4 + SRT export")

def require_binary(name):
    if shutil.which(name) is None:
        st.error(f"'{name}' was not found. Install FFmpeg and make sure ffmpeg/ffprobe are on your PATH.")
        st.stop()

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-5000:] or "Command failed")
    return p.stdout

def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip())

def srt_time(t):
    ms = max(0, int(round(float(t) * 1000)))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def ass_time(t):
    cs = max(0, int(round(float(t) * 100)))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02}:{s:02}.{cs:02}"

def clean_word(text):
    return str(text).replace("{", r"\{").replace("}", r"\}").replace("\n", " ").strip()

def make_srt(rows):
    blocks = []
    for i, r in enumerate(rows, 1):
        text = str(r["text"]).strip()
        if text:
            blocks.append(f"{i}\n{srt_time(r['start'])} --> {srt_time(r['end'])}\n{text}")
    return "\n\n".join(blocks) + "\n"

def make_ass(rows, font_size=66, placement="Lower-middle"):
    # ASS karaoke tags progressively highlight each word in a phrase.
    # Position is in a 1080x1920 portrait canvas.
    y = {"Lower-middle": 1430, "Center": 960, "Top": 300}[placement]
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Viral,Arial,{font_size},&H00FFFFFF,&H0000D7FF,&H00101010,&H80000000,-1,0,1,7,2,5,70,70,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for r in rows:
        words = re.findall(r"\S+", str(r["text"]).strip())
        if not words:
            continue
        start, end = float(r["start"]), float(r["end"])
        if end <= start:
            end = start + 0.15
        # distribute the phrase's total duration across words, weighted by character count
        weights = [max(1, len(re.sub(r"[^\w']", "", w))) for w in words]
        total = sum(weights)
        duration_cs = max(len(words), int(round((end - start) * 100)))
        vals = [max(1, int(duration_cs * w / total)) for w in weights]
        vals[-1] += duration_cs - sum(vals)
        vals[-1] = max(1, vals[-1])
        karaoke = " ".join(r"{\\kf" + str(v) + "}" + clean_word(w) for w, v in zip(words, vals))
        # Keep phrases readable and centered; karaoke fill changes word color in sync.
        events.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Viral,,0,0,0,,{{\\an5\\pos(540,{y})}}{karaoke}")
    return header + "\n".join(events) + "\n"

@st.cache_resource
def load_model(model_size):
    return WhisperModel(model_size, device="cpu", compute_type="int8")

require_binary("ffmpeg")
require_binary("ffprobe")

with st.sidebar:
    st.header("Caption settings")
    model_size = st.selectbox("Speech model", ["base", "small", "medium"], index=1,
                              help="Small is a good balance. Larger models are slower but may improve accuracy.")
    placement = st.selectbox("Caption placement", ["Lower-middle", "Center", "Top"], index=0)
    font_size = st.slider("Caption size", 42, 90, 66, 2)
    max_words = st.slider("Words per caption line", 3, 12, 7)
    st.caption("Default look: bold white text, dark outline, yellow karaoke-style word highlighting.")

uploaded = st.file_uploader("Upload a video file", type=["mp4", "mov", "mkv", "webm", "m4v"])
if uploaded:
    suffix = Path(uploaded.name).suffix or ".mp4"
    temp_dir = Path(tempfile.mkdtemp(prefix="shorts_captioner_"))
    source = temp_dir / f"source{suffix}"
    source.write_bytes(uploaded.getvalue())
    duration = probe_duration(source)
    st.video(uploaded)
    st.write(f"Video duration: **{duration:.1f} seconds**")

    c1, c2 = st.columns(2)
    with c1:
        clip_start = st.number_input("Clip start (seconds)", min_value=0.0, max_value=max(0.0, duration - 0.1),
                                     value=0.0, step=0.5)
    with c2:
        clip_end = st.number_input("Clip end (seconds)", min_value=0.1, max_value=max(0.1, duration),
                                   value=float(duration), step=0.5)
    if clip_end <= clip_start:
        st.warning("Clip end must be after clip start.")
        st.stop()

    if st.button("🎙️ Transcribe video", type="primary"):
        with st.spinner("Transcribing English speech and locating individual words…"):
            try:
                model = load_model(model_size)
                segments, info = model.transcribe(str(source), language="en", word_timestamps=True,
                                                   vad_filter=True, beam_size=5)
                words = []
                for seg in segments:
                    for w in (seg.words or []):
                        words.append({"start": float(w.start), "end": float(w.end),
                                      "text": w.word.strip()})
                # Clip timestamps to the selected excerpt and shift them to clip-relative time.
                clipped = []
                for w in words:
                    a, b = max(w["start"], clip_start), min(w["end"], clip_end)
                    if b > a:
                        clipped.append({"start": round(a - clip_start, 3),
                                        "end": round(b - clip_start, 3), "text": w["text"]})
                # Group words into short readable caption phrases, preserving word-level timing
                grouped = []
                buf = []
                for w in clipped:
                    buf.append(w)
                    joined = " ".join(x["text"] for x in buf)
                    if len(buf) >= max_words or re.search(r"[.!?]$", w["text"]) or len(joined) > 48:
                        grouped.append({"start": buf[0]["start"], "end": buf[-1]["end"], "text": joined})
                        buf = []
                if buf:
                    grouped.append({"start": buf[0]["start"], "end": buf[-1]["end"], "text": " ".join(x["text"] for x in buf)})
                st.session_state["caption_rows"] = grouped
                st.session_state["source_path"] = str(source)
                st.session_state["temp_dir"] = str(temp_dir)
                st.session_state["clip_start"] = float(clip_start)
                st.session_state["clip_end"] = float(clip_end)
                st.success(f"Transcription complete. Detected language: {info.language}. Review and edit captions below.")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

if "caption_rows" in st.session_state:
    st.subheader("✍️ Review and edit transcript")
    st.caption("Edit the start/end times and text directly in the table. Times are relative to the selected clip.")
    edited = st.data_editor(
        pd.DataFrame(st.session_state["caption_rows"]),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "start": st.column_config.NumberColumn("Start (s)", min_value=0.0, format="%.3f"),
            "end": st.column_config.NumberColumn("End (s)", min_value=0.0, format="%.3f"),
            "text": st.column_config.TextColumn("Caption text", width="large"),
        },
        key="caption_editor"
    )
    rows = []
    for _, row in edited.iterrows():
        try:
            a, b, text = float(row["start"]), float(row["end"]), str(row["text"]).strip()
            if text and b > a and a >= 0:
                rows.append({"start": a, "end": b, "text": text})
        except (ValueError, TypeError, KeyError):
            continue

    if rows:
        st.subheader("Caption preview")
        st.markdown("**" + rows[0]["text"].replace("<", "&lt;").replace(">", "&gt;") + "**")
        st.caption("Preview shows the first caption phrase; final styling is applied to the rendered video.")

    if st.button("🎬 Render captioned Short + SRT", type="primary", disabled=not rows):
        with st.spinner("Rendering vertical clip with synchronized captions…"):
            try:
                source = Path(st.session_state["source_path"])
                temp_dir = Path(st.session_state["temp_dir"])
                clip_start = st.session_state["clip_start"]
                clip_end = st.session_state["clip_end"]
                ass_path = temp_dir / "captions.ass"
                srt_path = temp_dir / "captions.srt"
                mp4_path = temp_dir / "captioned_short.mp4"
                ass_path.write_text(make_ass(rows, font_size, placement), encoding="utf-8")
                srt_path.write_text(make_srt(rows), encoding="utf-8")

                # Scale/crop to 9:16 and burn in ASS captions. Escape filter path for FFmpeg.
                escaped_ass = str(ass_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
                vf = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1,subtitles='{escaped_ass}'"
                cmd = ["ffmpeg", "-y", "-ss", str(clip_start), "-i", str(source),
                       "-t", str(clip_end - clip_start), "-vf", vf, "-c:v", "libx264",
                       "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "192k",
                       "-movflags", "+faststart", str(mp4_path)]
                run(cmd)
                st.success("Your captioned clip is ready.")
                with open(mp4_path, "rb") as f:
                    st.download_button("⬇️ Download captioned MP4", f, file_name="captioned_short.mp4",
                                       mime="video/mp4")
                with open(srt_path, "rb") as f:
                    st.download_button("⬇️ Download SRT subtitles", f, file_name="captions.srt",
                                       mime="application/x-subrip")
                st.video(str(mp4_path))
            except Exception as e:
                st.error(f"Rendering failed: {e}")
                st.info("Check that FFmpeg was installed with libass/subtitles support.")
else:
    st.info("Upload a video to begin. For best results, use a clear audio track and review the transcript before exporting.")
