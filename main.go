package main

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	_ "modernc.org/sqlite"
)

// ---------- GigaChat ----------

type GigaChatClient struct {
	clientID     string
	clientSecret string
	token        atomic.Value
	mu           sync.Mutex
	httpClient   *http.Client
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type tokenInfo struct {
	accessToken string
	expiresAt   int64
}

func NewGigaChatClient(clientID, clientSecret string) *GigaChatClient {
	return &GigaChatClient{
		clientID: clientID, clientSecret: clientSecret,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
			Transport: &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			},
		},
	}
}

func (g *GigaChatClient) getToken(ctx context.Context) (string, error) {
	if t, ok := g.token.Load().(*tokenInfo); ok && t.expiresAt > time.Now().UnixMilli() {
		return t.accessToken, nil
	}
	g.mu.Lock()
	defer g.mu.Unlock()
	if t, ok := g.token.Load().(*tokenInfo); ok && t.expiresAt > time.Now().UnixMilli() {
		return t.accessToken, nil
	}
	basic := base64.StdEncoding.EncodeToString([]byte(g.clientID + ":" + g.clientSecret))
	var data struct {
		AccessToken string `json:"access_token"`
		ExpiresAt   int64  `json:"expires_at"`
	}
	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		req, _ := http.NewRequestWithContext(ctx, "POST", "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
			bytes.NewBufferString("scope=GIGACHAT_API_PERS"))
		req.Header.Set("Authorization", "Basic "+basic)
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		req.Header.Set("Accept", "application/json")
		req.Header.Set("RqUID", newUUID())
		resp, err := g.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("oauth: %w", err)
			select {
			case <-ctx.Done():
				return "", lastErr
			case <-time.After(time.Duration(attempt) * time.Second):
			}
			continue
		}
		err = json.NewDecoder(resp.Body).Decode(&data)
		resp.Body.Close()
		if err != nil {
			lastErr = fmt.Errorf("oauth decode: %w", err)
			select {
			case <-ctx.Done():
				return "", lastErr
			case <-time.After(time.Duration(attempt) * time.Second):
			}
			continue
		}
		g.token.Store(&tokenInfo{accessToken: data.AccessToken, expiresAt: data.ExpiresAt})
		return data.AccessToken, nil
	}
	return "", lastErr
}

func (g *GigaChatClient) ChatStream(ctx context.Context, messages []chatMessage, tokenCh chan<- string, errCh chan<- error) {
	token, err := g.getToken(ctx)
	if err != nil {
		errCh <- err
		close(tokenCh)
		return
	}
	body := map[string]interface{}{
		"model": "GigaChat", "messages": messages,
		"temperature": 0.9, "max_tokens": 2048,
		"stream": true,
	}
	b, _ := json.Marshal(body)
	req, _ := http.NewRequestWithContext(ctx, "POST", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions", bytes.NewReader(b))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp, err := g.httpClient.Do(req)
	if err != nil {
		errCh <- fmt.Errorf("chat stream: %w", err)
		close(tokenCh)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		errCh <- fmt.Errorf("chat stream: HTTP %d: %s", resp.StatusCode, string(b))
		close(tokenCh)
		return
	}
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			tokenCh <- chunk.Choices[0].Delta.Content
		}
	}
	close(tokenCh)
	errCh <- scanner.Err()
}

func (g *GigaChatClient) Chat(ctx context.Context, messages []chatMessage) (string, error) {
	token, err := g.getToken(ctx)
	if err != nil {
		return "", err
	}
	body := map[string]interface{}{
		"model": "GigaChat", "messages": messages,
		"temperature": 0.9, "max_tokens": 2048,
	}
	b, _ := json.Marshal(body)

	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		req, _ := http.NewRequestWithContext(ctx, "POST", "https://gigachat.devices.sberbank.ru/api/v1/chat/completions", bytes.NewReader(b))
		req.Header.Set("Authorization", "Bearer "+token)
		req.Header.Set("Content-Type", "application/json")
		resp, err := g.httpClient.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("chat: %w", err)
			select {
			case <-ctx.Done():
				return "", lastErr
			case <-time.After(time.Duration(attempt) * time.Second):
			}
			continue
		}
		rb, _ := io.ReadAll(io.LimitReader(resp.Body, 65536))
		resp.Body.Close()
		if resp.StatusCode == 401 {
			// токен мог протухнуть — сбрасываем кэш
			g.token.Store(&tokenInfo{})
			token, err = g.getToken(ctx)
			if err != nil {
				return "", err
			}
			select {
			case <-ctx.Done():
				return "", fmt.Errorf("chat: context cancelled after 401")
			case <-time.After(time.Duration(attempt) * time.Second):
			}
			continue
		}
		if resp.StatusCode != http.StatusOK {
			lastErr = fmt.Errorf("chat: HTTP %d: %s", resp.StatusCode, string(rb))
			continue
		}
		var result struct {
			Choices []struct {
				Message struct{ Content string } `json:"message"`
			} `json:"choices"`
		}
		if err := json.Unmarshal(rb, &result); err != nil {
			lastErr = fmt.Errorf("chat decode: %w", err)
			select {
			case <-ctx.Done():
				return "", lastErr
			case <-time.After(time.Duration(attempt) * time.Second):
			}
			continue
		}
		if len(result.Choices) == 0 {
			return "", fmt.Errorf("no choices")
		}
		return result.Choices[0].Message.Content, nil
	}
	return "", lastErr
}

func newUUID() string {
	b := make([]byte, 16)
	rand.Read(b)
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

// ---------- SQLite память ----------

type MemoryDB struct {
	db *sql.DB
}

func NewMemoryDB(path string) (*MemoryDB, error) {
	db, err := sql.Open("sqlite", path+"?_journal_mode=WAL")
	if err != nil {
		return nil, err
	}
	db.Exec(`CREATE TABLE IF NOT EXISTS sessions (
		id TEXT PRIMARY KEY,
		name TEXT NOT NULL DEFAULT '',
		created_at TEXT NOT NULL DEFAULT (datetime('now'))
	)`)
	db.Exec(`CREATE TABLE IF NOT EXISTS messages (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		session_id TEXT NOT NULL REFERENCES sessions(id),
		role TEXT NOT NULL,
		content TEXT NOT NULL,
		created_at TEXT NOT NULL DEFAULT (datetime('now'))
	)`)
	db.Exec(`CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)`)
	return &MemoryDB{db: db}, nil
}

func (m *MemoryDB) GetOrCreateSession(id string) (string, error) {
	row := m.db.QueryRow("SELECT name FROM sessions WHERE id = ?", id)
	var name string
	if err := row.Scan(&name); err != nil {
		m.db.Exec("INSERT INTO sessions (id) VALUES (?)", id)
		return "", nil
	}
	return name, nil
}

func (m *MemoryDB) SaveMessage(sessionID, role, content string) error {
	_, err := m.db.Exec("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)", sessionID, role, content)
	return err
}

func (m *MemoryDB) GetHistory(sessionID string, limit int) ([]chatMessage, error) {
	rows, err := m.db.Query(
		"SELECT role, content FROM messages WHERE session_id = ? AND content != '' ORDER BY id DESC LIMIT ?", sessionID, limit,
	)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var msgs []chatMessage
	for rows.Next() {
		var m chatMessage
		rows.Scan(&m.Role, &m.Content)
		msgs = append(msgs, m)
	}
	// Разворачиваем: последние сообщения в хронологическом порядке
	for i, j := 0, len(msgs)-1; i < j; i, j = i+1, j-1 {
		msgs[i], msgs[j] = msgs[j], msgs[i]
	}
	return msgs, nil
}

func (m *MemoryDB) SetName(sessionID, name string) error {
	_, err := m.db.Exec("UPDATE sessions SET name = ? WHERE id = ?", name, sessionID)
	return err
}

// DeleteLastMessage удаляет последнее сообщение сессии (откат при ошибке)
func (m *MemoryDB) DeleteLastMessage(sessionID string) error {
	_, err := m.db.Exec(
		"DELETE FROM messages WHERE id = (SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1)",
		sessionID,
	)
	return err
}

// ClearMessages удаляет всю историю сессии
func (m *MemoryDB) ClearMessages(sessionID string) error {
	_, err := m.db.Exec("DELETE FROM messages WHERE session_id = ?", sessionID)
	return err
}

// ---------- Основной код ----------

var (
	db     *MemoryDB
	client *GigaChatClient
)

// TTS-кеш: ключ — SHA256 текста, значение — WAV байты
var ttsCache = struct {
	sync.RWMutex
	m map[string][]byte
}{m: make(map[string][]byte)}

func ttsKey(text string) string {
	h := sha256.Sum256([]byte(text))
	return fmt.Sprintf("%x", h[:12])
}

// currentVoice хранит выбранный голос (по умолчанию leonid)
var currentVoice string

// locationCtx — последнее известное местоположение пользователя
var locationCtx = struct {
	sync.RWMutex
	street    string
	city      string
	district  string
	poi       string
	updatedAt time.Time
}{}

// synthesize в фоне сохраняет WAV в кеш; повторные вызовы с тем же текстом ждут готовности
func synthesize(text, voice string) ([]byte, error) {
	key := ttsKey(text + "|" + voice)
	ttsCache.RLock()
	if wav, ok := ttsCache.m[key]; ok {
		ttsCache.RUnlock()
		return wav, nil
	}
	ttsCache.RUnlock()

	govoiceURL := os.Getenv("GOVOICE_URL")
	if govoiceURL == "" {
		govoiceURL = "http://localhost:8087"
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	body, _ := json.Marshal(map[string]string{"text": stripEmoji(text), "language": "ru", "voice": voice})
	httpReq, _ := http.NewRequestWithContext(ctx, "POST", govoiceURL+"/api/tts", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := (&http.Client{Timeout: 90 * time.Second}).Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("GoVoice недоступен: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("GoVoice: %s", string(b))
	}
	wav, _ := io.ReadAll(resp.Body)
	ttsCache.Lock()
	ttsCache.m[key] = wav
	ttsCache.Unlock()
	return wav, nil
}

func main() {
	addr := os.Getenv("ADDR")
	if addr == "" {
		addr = ":8098"
	}
	cid := os.Getenv("GIGACHAT_CLIENT_ID")
	cs := os.Getenv("GIGACHAT_CLIENT_SECRET")
	if cid == "" || cs == "" {
		log.Fatal("GIGACHAT_CLIENT_ID and GIGACHAT_CLIENT_SECRET required")
	}
	client = NewGigaChatClient(cid, cs)
	var err error
	db, err = NewMemoryDB("data/leonid.db")
	if err != nil {
		log.Fatalf("db: %v", err)
	}

	http.HandleFunc("/", handleRoot)
	http.HandleFunc("/api/chat", handleChat)
	http.HandleFunc("/api/chat/stream", handleChatStream)
	http.HandleFunc("/api/chat/clear", handleChatClear)
	http.HandleFunc("/api/history", handleHistory)
	http.HandleFunc("/api/name", handleName)
	http.HandleFunc("/api/voice", handleVoice)
	http.HandleFunc("/api/location", handleLocation)
	http.HandleFunc("/api/tts", handleTTS)
	http.HandleFunc("/api/tts-poll", handleTTSPoll)
	http.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(`{"status":"ok"}`))
	})

	log.Printf("AI-Leonid listening on %s", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

// ---------- Web ----------

func handleRoot(w http.ResponseWriter, r *http.Request) {
	path := r.URL.Path
	if path == "/favicon.ico" {
		w.Header().Set("Content-Type", "image/svg+xml")
		w.Write([]byte(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><circle cx="32" cy="32" r="30" fill="#58a6ff"/><text x="32" y="42" font-size="30" text-anchor="middle" fill="#0d1117" font-family="sans-serif">Л</text></svg>`))
		return
	}
	if path == "/" {
		b, err := os.ReadFile("index.html")
		if err != nil {
			http.Error(w, "index.html not found", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		w.Write(b)
		return
	}
	// Статические файлы (JS, CSS, jpg, png, etc.)
	http.FileServer(http.Dir(".")).ServeHTTP(w, r)
}

// ---------- API ----------

func getSessionID(r *http.Request) string {
	if s := r.URL.Query().Get("session"); s != "" {
		return s
	}
	if s := r.Header.Get("X-Session-ID"); s != "" {
		return s
	}
	// fallback — из куки
	c, err := r.Cookie("leonid_session")
	if err != nil {
		return newUUID()[:8]
	}
	return c.Value
}

// handleTTS — озвучивает текст голосом Леонида через GoVoice (клон голоса)
func handleTTS(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	var req struct {
		Text string `json:"text"`
		Voice string `json:"voice"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return
	}
	if strings.TrimSpace(req.Text) == "" {
		writeJSON(w, 400, map[string]string{"error": "text required"})
		return
	}

	wav, err := synthesize(req.Text, req.Voice)
	if err != nil {
		writeJSON(w, 502, map[string]string{"error": err.Error()})
		return
	}
	w.Header().Set("Content-Type", "audio/wav")
	w.Header().Set("Content-Length", fmt.Sprint(len(wav)))
	w.Write(wav)
}

// handleTTSPoll — выдаёт готовый WAV из кеша по ключу (без повторного синтеза)
func handleTTSPoll(w http.ResponseWriter, r *http.Request) {
	key := r.URL.Query().Get("key")
	if key == "" {
		writeJSON(w, 400, map[string]string{"error": "key required"})
		return
	}
	// Ждём готовности до 60 сек (синтез обычно 1-3 сек)
	deadline := time.Now().Add(60 * time.Second)
	for {
		ttsCache.RLock()
		wav, ok := ttsCache.m[key]
		ttsCache.RUnlock()
		if ok {
			w.Header().Set("Content-Type", "audio/wav")
			w.Header().Set("Content-Length", fmt.Sprint(len(wav)))
			w.Write(wav)
			return
		}
		if time.Now().After(deadline) {
			writeJSON(w, 404, map[string]string{"error": "wav not ready"})
			return
		}
		time.Sleep(100 * time.Millisecond)
	}
}

func handleName(w http.ResponseWriter, r *http.Request) {
	sid := getSessionID(r)
	if r.Method == "POST" {
		var req struct{ Name string `json:"name"` }
		json.NewDecoder(r.Body).Decode(&req)
		db.SetName(sid, req.Name)
		writeJSON(w, 200, map[string]string{"ok": "set"})
		return
	}
	name, _ := db.GetOrCreateSession(sid)
	writeJSON(w, 200, map[string]interface{}{"name": name})
}

// handleVoice — переключение голоса (leonid / optimus)
func handleVoice(w http.ResponseWriter, r *http.Request) {
	if r.Method == "POST" {
		var req struct {
			Voice string `json:"voice"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSON(w, 400, map[string]string{"error": err.Error()})
			return
		}
		if req.Voice != "leonid" && req.Voice != "optimus" && req.Voice != "chonishvili" {
			writeJSON(w, 400, map[string]string{"error": "voice must be leonid, optimus or chonishvili"})
			return
		}
		currentVoice = req.Voice
		log.Printf("Голос переключён на: %s", req.Voice)
		// Сбрасываем кеш TTS — старые WAV под другим голосом больше не нужны
		ttsCache.Lock()
		ttsCache.m = make(map[string][]byte)
		ttsCache.Unlock()
		writeJSON(w, 200, map[string]interface{}{"voice": currentVoice})
		return
	}
	writeJSON(w, 200, map[string]interface{}{"voice": currentVoice})
}

// handleLocation — принимает координаты браузера, возвращает адрес + POI через OSM
func handleLocation(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	var req struct {
		Lat float64 `json:"lat"`
		Lon float64 `json:"lon"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 15*time.Second)
	defer cancel()

	// 1. Nominatim — обратный геокодинг (адрес)
	addrURL := fmt.Sprintf("https://nominatim.openstreetmap.org/reverse?lat=%f&lon=%f&format=json&accept-language=ru",
		req.Lat, req.Lon)
	addrReq, _ := http.NewRequestWithContext(ctx, "GET", addrURL, nil)
	addrReq.Header.Set("User-Agent", "AI-Leonid/1.0")
	addrResp, err := http.DefaultClient.Do(addrReq)
	if err != nil {
		writeJSON(w, 502, map[string]string{"error": "Nominatim недоступен: " + err.Error()})
		return
	}
	defer addrResp.Body.Close()

	var addrData struct {
		DisplayName string `json:"display_name"`
		Address     struct {
			Road     string `json:"road"`
			City     string `json:"city"`
			Town     string `json:"town"`
			Village  string `json:"village"`
			District string `json:"suburb"`
			State    string `json:"state"`
		} `json:"address"`
	}
	if err := json.NewDecoder(addrResp.Body).Decode(&addrData); err != nil {
		writeJSON(w, 502, map[string]string{"error": "Nominatim decode: " + err.Error()})
		return
	}

	street := addrData.Address.Road
	city := addrData.Address.City
	if city == "" {
		city = addrData.Address.Town
	}
	if city == "" {
		city = addrData.Address.Village
	}
	district := addrData.Address.District

	// 2. Overpass API — ближайшие магазины, аптеки, банкоматы
	poiQuery := fmt.Sprintf(`[out:json];
(
  node["shop"](around:500,%f,%f);
  node["amenity"="pharmacy"](around:500,%f,%f);
  node["amenity"="atm"](around:300,%f,%f);
  node["amenity"="bank"](around:500,%f,%f);
  node["amenity"="hospital"](around:1000,%f,%f);
  node["amenity"="clinic"](around:500,%f,%f);
  node["amenity"="parking"](around:300,%f,%f);
  node["amenity"="bus_station"](around:500,%f,%f);
);
out 8;`,
		req.Lat, req.Lon, req.Lat, req.Lon, req.Lat, req.Lon,
		req.Lat, req.Lon, req.Lat, req.Lon, req.Lat, req.Lon,
		req.Lat, req.Lon, req.Lat, req.Lon)

	poiURL := "https://overpass-api.de/api/interpreter"
	poiReq, _ := http.NewRequestWithContext(ctx, "POST", poiURL, bytes.NewBufferString("data="+url.QueryEscape(poiQuery)))
	poiReq.Header.Set("User-Agent", "AI-Leonid/1.0")
	poiReq.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	poiResp, err := http.DefaultClient.Do(poiReq)
	if err != nil {
		log.Printf("Overpass error: %v", err)
		writeJSON(w, 200, map[string]interface{}{
			"street": street, "city": city, "district": district, "poi": []string{},
		})
		return
	}
	defer poiResp.Body.Close()

	var overpass struct {
		Elements []struct {
			Type string `json:"type"`
			Tags struct {
				Name   string `json:"name"`
				Shop   string `json:"shop"`
				Amenity string `json:"amenity"`
			} `json:"tags"`
		} `json:"elements"`
	}
	raw, _ := io.ReadAll(io.LimitReader(poiResp.Body, 32768))
	log.Printf("Overpass raw (first 200): %s", string(raw[:min(len(raw), 200)]))
	if err := json.Unmarshal(raw, &overpass); err != nil {
		log.Printf("Overpass decode error: %v", err)
		writeJSON(w, 200, map[string]interface{}{
			"street": street, "city": city, "district": district, "poi": []string{},
		})
		return
	}
	log.Printf("Overpass elements: %d", len(overpass.Elements))

	var pois []string
	for _, el := range overpass.Elements {
		name := el.Tags.Name
		if name == "" {
			if el.Tags.Shop != "" {
				name = el.Tags.Shop
			} else if el.Tags.Amenity != "" {
				name = el.Tags.Amenity
			}
		}
		if name != "" {
			pois = append(pois, name)
		}
	}
	if len(pois) > 5 {
		pois = pois[:5]
	}
	poiJSON, _ := json.Marshal(pois)

	// Сохраняем в кеш
	locationCtx.Lock()
	locationCtx.street = street
	locationCtx.city = city
	locationCtx.district = district
	locationCtx.poi = string(poiJSON)
	locationCtx.updatedAt = time.Now()
	locationCtx.Unlock()

	writeJSON(w, 200, map[string]interface{}{
		"street": street, "city": city, "district": district, "poi": pois,
	})
}

func handleHistory(w http.ResponseWriter, r *http.Request) {
	sid := getSessionID(r)
	db.GetOrCreateSession(sid)
	history, _ := db.GetHistory(sid, 50)
	writeJSON(w, 200, map[string]interface{}{"history": history})
}

// handleChatClear — очищает историю сессии (кнопка «Очистить»)
func handleChatClear(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	sid := getSessionID(r)
	db.ClearMessages(sid)
	writeJSON(w, 200, map[string]string{"ok": "cleared"})
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	sid := getSessionID(r)
	name, _ := db.GetOrCreateSession(sid)

	var req struct {
		Message string `json:"message"`
		Voice   string `json:"voice"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return
	}

	if req.Voice == "leonid" || req.Voice == "optimus" || req.Voice == "chonishvili" {
		currentVoice = req.Voice
	}

	// Ключевые слова в тексте — гарантированно переключаем персонажа
	if v := detectKeywordVoice(req.Message); v != "" {
		currentVoice = v
	}

// Сохраняем сообщение пользователя
	db.SaveMessage(sid, "user", req.Message)

	// Формируем контекст: системный промпт + только последние 6 сообщений
	history, _ := db.GetHistory(sid, 6)
	messages := []chatMessage{
		{Role: "system", Content: buildPrompt(name)},
	}
	messages = append(messages, history...)

	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()

	reply, err := client.Chat(ctx, messages)
	if err != nil {
		writeJSON(w, 500, map[string]string{"error": err.Error()})
		return
	}

	// GigaChat иногда игнорирует запрет эмодзи — вычищаем на сервере
	reply = stripEmoji(reply)

// Пустой ответ — повторяем запрос один раз
	if strings.TrimSpace(reply) == "" {
		reply, err = client.Chat(ctx, messages)
		if err == nil {
			reply = stripEmoji(reply)
		}
	}

	if strings.TrimSpace(reply) == "" {
		db.DeleteLastMessage(sid)
		writeJSON(w, 502, map[string]string{"error": "GigaChat вернул пустой ответ, попробуй ещё раз"})
		return
	}

	db.SaveMessage(sid, "assistant", reply)

	// Синтез голоса параллельно, ждём до 12 сек, чтобы текст+аудио пришли одновременно
	var ttsBase64 string
	if os.Getenv("VOICE") != "off" {
		ttsDone := make(chan struct{})
		go func() {
			wav, err := synthesize(reply, currentVoice)
			if err == nil && len(wav) > 0 {
				ttsBase64 = base64.StdEncoding.EncodeToString(wav)
			}
			close(ttsDone)
		}()
		select {
		case <-ttsDone:
		case <-time.After(12 * time.Second):
		}
	}

	writeJSON(w, 200, map[string]interface{}{"reply": reply, "tts": ttsBase64})
}

type sseEvent struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	Audio string `json:"audio,omitempty"`
	Error string `json:"error,omitempty"`
}

func splitSentences(s string) []string {
	var res []string
	start := 0
	for i, r := range s {
		if r == '.' || r == '!' || r == '?' {
			res = append(res, s[start:i+1])
			start = i + 1
		}
	}
	if start < len(s) {
		res = append(res, s[start:])
	}
	return res
}

func synthesizeSentence(text string, ch chan<- sseEvent, voice string) {
	wav, err := synthesize(text, voice)
	if err != nil || len(wav) == 0 {
		return
	}
	ch <- sseEvent{Type: "audio", Audio: base64.StdEncoding.EncodeToString(wav)}
}

// handleChatStream — SSE-стрим: текст по токенам + голос по предложениям
func handleChatStream(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		writeJSON(w, 405, map[string]string{"error": "POST only"})
		return
	}
	sid := getSessionID(r)
	name, _ := db.GetOrCreateSession(sid)

	var req struct {
		Message string `json:"message"`
		Voice   string `json:"voice"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, 400, map[string]string{"error": err.Error()})
		return
	}

	if req.Voice == "leonid" || req.Voice == "optimus" || req.Voice == "chonishvili" {
		currentVoice = req.Voice
	}

	// Ключевые слова в тексте — гарантированно переключаем персонажа
	if v := detectKeywordVoice(req.Message); v != "" {
		currentVoice = v
	}

	db.SaveMessage(sid, "user", req.Message)

	history, _ := db.GetHistory(sid, 6)
	messages := []chatMessage{
		{Role: "system", Content: buildPrompt(name)},
	}
	messages = append(messages, history...)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeJSON(w, 500, map[string]string{"error": "streaming unsupported"})
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), 90*time.Second)
	defer cancel()

	tokenCh := make(chan string, 64)
	errCh := make(chan error, 1)
	go client.ChatStream(ctx, messages, tokenCh, errCh)

	// Канал для отправки событий (текст + аудио)
	eventCh := make(chan sseEvent, 32)

	var fullText strings.Builder
	buf := strings.Builder{}
	voiceOn := os.Getenv("VOICE") != "off"
	voice := currentVoice
	if voice == "" {
		voice = "leonid"
	}

	// Запускаем поток чтения токенов GigaChat
	go func() {
		defer close(eventCh)
		for {
			select {
			case <-ctx.Done():
				return
			case tok, ok := <-tokenCh:
				if !ok {
					// Поток закончился — отправляем остаток
					remaining := strings.TrimSpace(buf.String())
					if remaining != "" {
						remaining = stripEmoji(remaining)
						fullText.WriteString(remaining + " ")
						eventCh <- sseEvent{Type: "text", Text: remaining}
						if voiceOn {
							synthesizeSentence(remaining, eventCh, voice)
						}
					}
					final := strings.TrimSpace(fullText.String())
					if final == "" {
						eventCh <- sseEvent{Type: "error", Error: "пустой ответ"}
						return
					}
					db.SaveMessage(sid, "assistant", final)
					eventCh <- sseEvent{Type: "done", Text: final}
					return
				}
				buf.WriteString(tok)
				// Пробуем вырезать предложение
				s := buf.String()
				parts := splitSentences(s)
				if len(parts) > 1 {
					// Всё кроме последнего — готовые предложения
					ready := strings.Join(parts[:len(parts)-1], "")
					ready = stripEmoji(strings.TrimSpace(ready))
					if ready != "" {
						fullText.WriteString(ready + " ")
						eventCh <- sseEvent{Type: "text", Text: ready}
						if voiceOn {
							synthesizeSentence(ready, eventCh, voice)
						}
					}
					buf.Reset()
					buf.WriteString(parts[len(parts)-1])
				}
			}
		}
	}()

	// Пишем события клиенту (один поток, без гонок)
	for ev := range eventCh {
		json.NewEncoder(w).Encode(ev)
		flusher.Flush()
	}
}

func buildPrompt(name string) string {
	personaName := "Леонид"
	personaDesc := "живой и общительный друг-собеседник"
	switch currentVoice {
	case "optimus":
		personaName = "Оптимус Прайм"
		personaDesc = "легендарный лидер автоботов из вселенной Transformers, мудрый и благородный защитник"
	case "chonishvili":
		personaName = "Риддик"
		personaDesc = "Ричард Б. Риддик — опасный хрипловатый антигерой, последний из своего рода, говорит сдержанно и угрюмо"
	}
	p := fmt.Sprintf(`Ты — %s. Ты %s.
Твои особенности:
- Отвечаешь на "ты", коротко и живо, как в разговоре.
- Используешь разговорные фразы и иногда легкий юмор.
- Никогда не используешь смайлики, эмодзи и графические символы — только буквы и знаки препинания.
- Спрашиваешь в ответ, поддерживаешь беседу.
- Не бываешь канцелярским и сухим.
- Если собеседник говорит о кибербезопасности или IT — подхватываешь, потому что это твоя тема.
- Называешь собеседника по имени, если знаешь.
- Если не знаешь ответа, говоришь честно: "Не знаю, но давай разберёмся вместе".
- Пишешь короткими предложениями (1-3 на ответ).
- Всегда представляешься и ведёшь себя как %[1]s, никогда не называешь себя другим именем, кроме %[1]s.`, personaName, personaDesc)
	if name != "" {
		p += "\nИмя собеседника: " + name + ". Обращайся к нему по имени."
	}
	// Местоположение пользователя (если известно)
	locationCtx.RLock()
	hasLoc := locationCtx.city != "" || locationCtx.street != ""
	locStreet := locationCtx.street
	locCity := locationCtx.city
	locDistrict := locationCtx.district
	locPOI := locationCtx.poi
	locationCtx.RUnlock()
	if hasLoc {
		p += "\n\nТекущее местоположение собеседника:"
		if locStreet != "" {
			p += "\n- Улица: " + locStreet
		}
		if locCity != "" {
			p += "\n- Город/населённый пункт: " + locCity
		}
		if locDistrict != "" {
			p += "\n- Район: " + locDistrict
		}
		if locPOI != "" && locPOI != "[]" && locPOI != "null" {
			p += "\n- Рядом есть: " + locPOI
		}
		p += "\nИспользуй эти сведения, если собеседник спрашивает про местность, маршруты или ближайшие объекты."
	}
	return p
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(v)
}

// detectKeywordVoice — определяет персонажа по ключевым словам в тексте сообщения
func detectKeywordVoice(text string) string {
	t := strings.ToLower(text)
	// Проверяем в порядке приоритета (самые специфичные первыми)
	if strings.Contains(t, "риддик") || strings.Contains(t, "ричард") {
		return "chonishvili"
	}
	if strings.Contains(t, "оптимус") || strings.Contains(t, "optimus") || strings.Contains(t, "прайм") || strings.Contains(t, "prime") || strings.Contains(t, "автобот") {
		return "optimus"
	}
	if strings.Contains(t, "леонид") || strings.Contains(t, "лёня") || strings.Contains(t, "леня") {
		return "leonid"
	}
	return ""
}

// stripEmoji удаляет эмодзи, спецсимволы и лишнюю пунктуацию — XTTS не умеет их синтезировать
func stripEmoji(s string) string {
	var b strings.Builder
	for _, r := range s {
		switch {
		case r == '\n' || r == '\t' || r == '\r':
			b.WriteRune(' ')
		case r >= 0x1F600 && r <= 0x1F64F: // emoticons
		case r >= 0x1F300 && r <= 0x1F5FF: // misc symbols & pictographs
		case r >= 0x1F680 && r <= 0x1F6FF: // transport
		case r >= 0x1F1E6 && r <= 0x1F1FF: // flags
		case r >= 0x2600 && r <= 0x27BF:   // misc symbols
		case r >= 0xFE00 && r <= 0xFE0F:   // variation selectors
		case r >= 0x200D:                  // zero-width joiner etc
		case r < 0x20:                     // control chars
		case r == '*' || r == '`' || r == '~' || r == '_': // markdown — XTTS читает их как звуки
		case r == '|' || r == '\\' || r == '#':
		default:
			b.WriteRune(r)
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}