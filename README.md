# AI-Agents

Проект содержит трёх AI-помощников-собеседников.

- **Леонид** — безопасник по ИБ
- **Оптимус** — специалист по защите галактики
- **Риддик** — даёт нужные советы

## Возможности

- Три персонажа с уникальными клонированными голосами (XTTS v2)
- Переключение персонажа кнопками в сайдбаре или по ключевым словам: «Леонид», «Оптимус», «Риддик»
- Голосовой ввод (кнопка микрофон), отправка после паузы 3 сек
- История диалога в SQLite (последние 6 сообщений в контексте)
- Геолокация: персонажи знают местоположение и ближайшие объекты (OpenStreetMap)
- Веб-интерфейс с тёмной темой, синхронизация текста и голоса

## Скачивание

```bash
git clone https://github.com/LeonidUspeshny/Ai-agents
cd Ai-agents
```

## Требования

| Компонент | Версия | Зачем |
|---|---|---|
| Go | 1.25+ | Сборка сервера `ai-leonid.exe` |
| Python | 3.11 (64-bit) | GoVoice (TTS/STT) |
| GPU NVIDIA с CUDA | 8+ GB VRAM (RTX 3060 и лучше) | Синтез голоса XTTS v2 |
| Microsoft C++ Build Tools | 2022 (Workload VCTools) | Сборка библиотеки TTS |
| Ключи GigaChat API | Client ID / Secret | Доступ к модели GigaChat |

> Без GPU проект тоже запустится: голос будет работать на CPU (медленнее, ~30-60 сек на фразу), либо можно отключить голос (`VOICE=off`).

## Установка

### 1. GigaChat API (бесплатно)

1. Зарегистрируйся на https://developers.sber.ru (SberDevices)
2. Создай приложение, получи `Client ID` и `Client Secret`
3. Эти ключи понадобятся в шаге 4 — они не хранятся в репозитории

### 2. Go-сервер

```bash
go mod tidy
go build -o ai-leonid.exe .
```

### 3. GoVoice (голос)

```bash
python -m venv govoice/govoice-env
govoice\govoice-env\Scripts\activate      # Windows
# или: source govoice/govoice-env/bin/activate   # Linux/macOS

pip install "torch==2.4.1+cu121" "torchaudio==2.4.1+cu121" "torchvision==0.19.1+cu121" --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-go.txt
pip install "TTS==0.22.0" "transformers==4.46.3" "tokenizers==0.20.3" "faster-whisper" "sounddevice" "pynput"
```

### 4. Голосовые сэмплы

Для каждого персонажа нужен WAV-сэмпл голоса (10-60 сек чистой речи), положи в `govoice/data/`:

| Персонаж | Файл |
|---|---|
| Леонид | `govoice/data/speaker.wav` |
| Оптимус | `govoice/data/optimus_prime.wav` |
| Риддик | `govoice/data/chonishvili.wav` |

Без сэмплов персонажи будут отвечать стандартным голосом XTTS.

### 5. Запуск

Создай `run.bat` (или запускай команды вручную):

```bat
set COQUI_TOS_AGREED=1
set GIGACHAT_CLIENT_ID=ТВОЙ_CLIENT_ID
set GIGACHAT_CLIENT_SECRET=ТВОЙ_CLIENT_SECRET
set ADDR=:8098
set GOVOICE_URL=http://localhost:8087

start "govoice" govoice\govoice-env\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8087 --app-dir govoice
start /B ai-leonid.exe
timeout /t 5
start http://localhost:8098
```

Открой `http://localhost:8098` — всё готово.

## Структура

```
Ai-agents/
├── main.go            # Go-сервер: чат, GigaChat, геолокация, история
├── index.html         # Веб-интерфейс
├── go.mod             # Go-зависимости
├── govoice/
│   ├── server.py      # Python-сервер: XTTS (голос) + Whisper (STT)
│   └── data/          # Голосовые сэмплы (не в git)
└── govoice-hotkey.py  # (заморожено) голосовой помощник с хоткеем Ctrl+Z
```

## Примечание по приватности

- Свои ключи GigaChat никому не передавай — они только в твоём `run.bat` и в переменных окружения, `.gitignore` исключает их из коммитов.
- Геолокация запрашивается браузером только при нажатии/активации персонажа, координаты хранятся локально (кеш 2 мин).