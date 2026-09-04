# pi-mcp — Code aus der Sitzung vom 03.09.2026

Snapshot des Codes, der auf dem Raspberry Pi 5 (`192.168.178.35`) läuft, plus
der Diagnose-Werkzeuge, die in dieser Sitzung entstanden sind.

**Das hier ist eine Kopie, nicht die Quelle.** Produktiv läuft alles unter
`/home/seb/stereo-mcp/` auf dem Pi. Änderungen hier wirken erst, wenn sie
zurückkopiert werden.

## Was der Server tut

Ein MCP-Endpunkt unter `http://192.168.178.35:8000/mcp` (streamable HTTP)
stellt Stereokamera **und** Ultraschallsensor als Tools bereit. Client ist
LM Studio („Bionic") auf der Windows-Box mit Qwen3.8-27B.

| Datei | Rolle |
|---|---|
| `server.py` | MCP-Server, 8 Tools. In dieser Sitzung um die Ultraschall-Tools erweitert. |
| `ultraschall.py` | HC-SR04 über `lgpio`, mit Lock und 70 ms Pulsabstand. |
| `systemd/*.service` | Die beiden Units, in dieser Sitzung angelegt. |
| `tools/*.py` | Diagnose-Werkzeuge, siehe unten. |

Nicht mitkopiert, weil unverändert: `vision.py` (OpenCV + YOLOv4-tiny + SGBM)
und `stream.py` (MJPEG-Streamer). Liegen auf dem Pi.

## Die zwei Dienste

| Unit | Port | Zweck |
|---|---|---|
| `stereo-mcp.service` | 8000 | Der MCP-Endpunkt |
| `stereo-stream.service` | 8001 | MJPEG-Streamer, hält beide Sensoren offen |

Eine Kamera kann nur von *einem* Prozess gehalten werden. Der Streamer besitzt
beide Sensoren dauerhaft, deshalb käme `rpicam-still` nicht an sie heran — der
MCP-Server holt seine Bilder darum von `/snapshot/camN.jpg` und fällt nur
zurück, wenn der Streamer weg ist. Nebeneffekt: Aufnahmelatenz 1,4 s → 0,1 s.

Daher hat `stereo-mcp` ein weiches `Wants=` auf `stereo-stream` plus
`ExecStartPre=/bin/sleep 2` — weich, weil der Server auch ohne Streamer
funktioniert.

## Werkzeuge

Alle drei laufen **auf dem Pi** und brauchen dessen venv:

```bash
~/stereo-mcp/venv/bin/python <script>
```

- **`tools/mcp_client.py`** — Tools auflisten und aufrufen.
  `list` · `call ultraschall_abstand` · `call calibrate_stereo left_camera=1`
- **`tools/eye_order_check.py`** — bestimmt per Parallaxe, welche Kamera das
  linke Auge ist. Nach jedem Umbau des Rigs laufen lassen.
- **`tools/ultraschall_stability.py`** — Messreihe; unterscheidet Sensorfehler
  von Szenenunruhe.

## Fallstricke, die Zeit gekostet haben

- **`RPi.GPIO` funktioniert auf dem Pi 5 nicht** (GPIO sitzt im RP1). `lgpio`
  nutzen. `pigpio` scheidet ebenfalls aus.
- **MCP-SDK ist 2.x:** `FastMCP` → `MCPServer`; `host`/`port` gehören an
  `run()`, nicht in den Konstruktor. Client-seitig heißt es
  `streamable_http_client`, nicht `streamablehttp_client`.
- **Synchrone Tool-Handler laufen in v2 in einem Worker-Thread** — Hardware
  braucht deshalb ein Lock. `Ultraschall` bringt eines mit.
- **DNS-Rebinding-Schutz ist per Default an, mit leerer Allowlist**, was jeden
  LAN-Client abweist. Die IP muss in `allowed_hosts`, blank *und* mit `:8000`.
- **Der laufende Dienst hält GPIO23/24 exklusiv.** Ein zweites Skript, das die
  Pins selbst öffnet, bekommt „GPIO busy" — Tests deshalb über den Endpunkt.
- **`pkill` braucht das richtige Muster.** Wird der Server als
  `./venv/bin/python server.py` gestartet, greift ein `pkill -f` auf den
  absoluten Pfad **nicht**. Das hat Port 8000 blockiert und die Unit in einen
  Restart-Loop geschickt. Auf `venv/bin/python server.py` matchen.

## Augenreihenfolge

**Kamera 1 ist das linke Auge, Kamera 0 das rechte** — gemessen, nicht
angenommen (nah +197 px, fern −56 px relative Disparität).

Die Auto-Inferenz von `calibrate_stereo` liegt hier **falsch** (meldet Kamera 0),
kennzeichnet sich aber selbst als `INCONCLUSIVE`. Immer explizit
`left_camera=1` übergeben.

## Kalibrierung

Aktueller Stand `USABLE`: Vertikalversatz +47,0 px, Residuum 0,8 px,
Horizontalversatz bei Unendlich −59,9 px.

Gegen den Ultraschall validiert: Stereo 0,33 m vs. Ultraschall 0,278 m am selben
Objekt — innerhalb der angegebenen ±15 %.

**Grenzen:** Die Tiefenabdeckung lag bei nur 6 % der Pixel, weil die Szene eine
glänzende Tischplatte und eine waagerecht gelattete Wand enthielt — beides
liefert kaum senkrechte Kanten. Und große `depth_grid`-Werte (13 m, 26 m) sind
bei 60 mm Basisbreite Sub-Pixel-Disparitäten: sie bedeuten „weit/unbekannt",
nicht eine Messung.

## Sicherheit

Bewusste Entscheidung des Nutzers: Das System läuft nur im LAN, es gibt **keine
Firewall** und Passwort-Login über SSH bleibt aktiv.
