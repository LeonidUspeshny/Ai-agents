# Голосовой ассистент ИИ — Леонид (глобальный хоткей Ctrl+Win+X)
# Поток: хоткей -> запись с микрофона -> STT (Whisper) -> GigaChat -> TTS -> проигрывание
import io
import time
import wave
import base64
import json
import os
import sys
import threading
import tempfile
import urllib.request
import sounddevice as sd
import numpy as np
from pynput import keyboard

# Атомарный именованный мьютекс — защита от дублей (Windows)
import ctypes
_kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
_MUTEX_NAME = "AILeonidHotkeyMutex"
_hmutex = _kernel32.CreateMutexW(None, False, _MUTEX_NAME)
if not _hmutex or ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
    sys.exit(0)

AI_URL = "http://localhost:8098"
GOVOICE_URL = "http://localhost:8087"
SESSION = "hotkey"
SAMPLE_RATE = 16000

recording = False
frames = []


def play_wav_bytes(data: bytes):
    """Проигрывает WAV из байтов."""
    tmp = tempfile.mktemp(suffix=".wav")
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        import winsound
        winsound.PlaySound(tmp, winsound.SND_FILENAME)
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


def post_json(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_wav(url, wav_bytes):
    boundary = "----HotkeyBoundary"
    body = b"--" + boundary.encode()
    body += b'\r\nContent-Disposition: form-data; name="file"; filename="voice.wav"\r\n'
    body += b"Content-Type: audio/wav\r\n\r\n"
    body += wav_bytes
    body += b"\r\n--" + boundary.encode() + b"--\r\n"
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "multipart/form-data; boundary=" + boundary})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def handle_hotkey():
    print("[hotkey] распознаю речь... (говори)")
    wav_bytes = record_to_wav()
    if wav_bytes is None:
        return
    print("[hotkey] распознал, отправляю в STT...")
    try:
        stt = post_wav(GOVOICE_URL + "/api/stt", wav_bytes)
        text = stt.get("text", "").strip()
    except Exception as e:
        print("[hotkey] STT ошибка:", e)
        return
    if not text:
        print("[hotkey] речь не распознана")
        return
    print("[hotkey] вопрос:", text)
    try:
        data = post_json(AI_URL + "/api/chat?session=" + SESSION, {"message": text})
    except Exception as e:
        print("[hotkey] chat ошибка:", e)
        return
    reply = data.get("reply", "")
    tts_b64 = data.get("tts", "")
    print("[hotkey] ответ:", reply)
    if tts_b64:
        try:
            play_wav_bytes(base64.b64decode(tts_b64))
        except Exception as e:
            print("[hotkey] воспроизведение ошибка:", e)


def record_to_wav():
    """Ждёт начала речи, записывает до паузы, возвращает WAV-байты."""
    global frames
    frame_len = int(SAMPLE_RATE * 0.25)
    recording_frames = []
    silence_q = []
    started = False
    t_start = time.time()
    max_dur = 15
    print("  слушаю...")
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        while time.time() - t_start < max_dur:
            block, _ = stream.read(frame_len)
            amp = float(np.max(np.abs(block))) if len(block) else 0.0
            if amp > 0.02:
                started = True
                silence_q = []
                recording_frames.append(block)
            elif started:
                silence_q.append(block)
                if len(silence_q) == 1:
                    recording_frames.append(block)
                if len(silence_q) >= 12:  # ~3 сек тишины -> стоп
                    break
            if started and len(recording_frames) > 0:
                el = 0
                for blk in recording_frames:
                    el += len(blk)
                if el / SAMPLE_RATE > max_dur:
                    break
    if not recording_frames:
        print("  (не услышал речь)")
        return None
    audio = np.concatenate(recording_frames, axis=0).flatten()
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    print(f"  записано {len(audio)/SAMPLE_RATE:.1f}s")
    return buf.getvalue()


def on_activate():
    threading.Thread(target=handle_hotkey, daemon=True).start()


def main():
    print("Голосовой ассистент ИИ — Леонид")
    print("Хоткей: Ctrl+Z (нажми, скажи фразу, получи ответ голосом)")
    print("Ctrl+C для выхода")
    hotkey = keyboard.HotKey(keyboard.HotKey.parse("<ctrl>+z"), on_activate)
    with keyboard.Listener(on_press=hotkey.press, on_release=hotkey.release) as listener:
        listener.join()


if __name__ == "__main__":
    main()