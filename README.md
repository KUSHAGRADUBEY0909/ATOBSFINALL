# ATOBS — Bengaluru Bus Occupancy, Tracking & Safety

A full-stack local web app built on the ATOBS BMTC dataset (2,045 routes, 5,348 stops, 63,187 stop-level trip observations).

| Feature | What it does |
|---|---|
| Route search | Search any route by number or place. |
| **Bus timings** | Pick any two places and get every direct bus between them, in either direction: the dataset timetable plus a bus every **10 min at peak** (07–11, 17–21) and every **15 min otherwise** (05:30–23:00). Times are traffic-adjusted and the list refreshes live. Each bus shows its expected occupancy and a seat map. |
| Route page | Stops, timetable, map, live buses, a **crowd forecast** per stop for any hour, and a **predicted seat map** at any stop. |
| **Occupancy %** | Whole-number percentage everywhere (e.g. **66%**), based on the 40 seats of a BMTC ordinary bus. Shown with a **40-seat map**: green = free, red = occupied, yellow dots = standing. |
| Crowd forecast (ML) | Gradient-boosting model trained on `bus_occupancy` from the batch workbooks. Held-out-route accuracy: **MAE 4.2 points, R² 0.87**. |
| **Camera passenger counting** | The operator's camera (webcam / video file / photo) is analysed by YOLOv8. Each person is tracked, and crossing the door line counts a boarding or alighting. The on-board count updates the bus's seat map everywhere in the app. |
| **Camera simulation page** | Plays test clips with the detections overlaid and compares the camera count with the **actual** count (ground truth) frame by frame. Shows accuracy, a count-over-time chart, and the actual vs. estimated seat maps. Staff can upload their own bus videos and add hand-checked counts. |
| **Verified route start/end stops** | Route start and end stops are checked against published map coordinates during import. Wrong ones are corrected and name variants merged; every change is logged (Control room → *Route data quality*, `data/terminal_audit.csv`). |
| **Emergency alert** | Red **EMERGENCY** button fixed at the **bottom-left of every page**: emergency type, description, route/bus, contact, optional GPS. Returns a reference number the passenger can track. |
| Control room | Alerts by severity with sound + banner, status workflow with notes, live fleet occupancy, counting sessions, data-quality log. |
| Auth & roles | PBKDF2 password hashing, JWT httpOnly cookie. Roles: passenger, operator, admin. |

## Run it

Requires **Python 3.10+**.

```bash
# Windows
run.bat
# macOS / Linux
./run.sh
```

Or manually:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
python -m uvicorn backend.app:app --port 8000
```

Open **http://localhost:8000**. The database, trained model, detector weights, simulation videos and their evaluation results are all included.

### Demo accounts

| Role | Email | Password |
|---|---|---|
| Admin | admin@atobs.local | admin123 |
| Operator (conductor) | operator@atobs.local | operator123 |
| Passenger | passenger@atobs.local | passenger123 |

## Live headcount demo (for presentations)

Open **Live demo** in the top bar. No login is needed.

1. **Drop a photo or video** onto the page, or click *Choose file*. It accepts JPG/PNG photos and MP4/WebM/MOV videos, including iPhone HEVC videos.
2. **Photo**: the count appears in under a second, with a numbered box on every person so anyone can count along.
3. **Video**: press **Analyse video** to step through the clip (every 0.5 / 1 / 2 s) and watch the count update frame by frame, or **Play with live count** to count while it plays normally. The chart shows the headcount over time; click a point to jump back to that frame.
4. The **40-seat bus map** and **occupancy %** follow the count.
5. **Is the count correct?** Pause on a frame, type the real number of people and press *Check*. The page shows the accuracy for that frame, and the history table keeps every run and check.

*Accurate* mode (default) also counts on 3×2 zoomed tiles, so people at the back are found. *Fast* is a single pass. Videos the browser can't play are processed on the server automatically, with each frame shown as it's counted.

Sample photos and videos (real footage from the Intel IoT DevKit sample videos, plus the simulated bus) are on the page for a quick start. Tuning results on photos with known counts:

| Photo | Real count | Counted |
|---|---|---|
| People walking in (real) | 3 | 3 |
| Group at entrance (real) | 3 | 3 |
| Seated rows (real) | 4 | 4 |
| Half-full bus (simulated) | 21 | 15 |
| Packed bus (simulated) | 44 | 25 |

A single front camera undercounts a packed bus because the back rows are hidden behind the heads in front. Larger YOLOv8 models (s, m) didn't fix this and were 3–7× slower. For full buses use the door-line count (see Camera simulation), which is 98% accurate.

**Tips for a live review:** test your clips beforehand. The camera should be at the front of the bus looking back, with good light. Videos of 10–30 s at 720p analyse in well under a minute.

## Camera simulation — is the count correct?

Open **Camera simulation** in the top bar. Three clips are included, each with ground truth:

| Clip | Camera position | How the true count is known | Result |
|---|---|---|---|
| **Bus cabin simulation** (40 seats, 9 stops, up to 44 people) | Front of the bus, above the windscreen, looking back down the aisle | The simulator places every passenger, so the exact count, and which seats are taken, is known for all 1,408 frames | **Door count 98% accurate**: on-board count exact 58% of the time, within ±1 person 90%, ±2 96%. 58/60 boardings and 52/57 alightings detected. Cabin headcount alone: 61% (undercounts a full bus). |
| Real footage: seated rows, front camera (Intel `classroom.mp4`) | Fixed front camera facing seated rows | Counted by hand once per second (33 checks) | Headcount **93%**: exact 73%, within ±1 100% |
| Real footage: entrance, ceiling camera (Intel `people-detection.mp4`) | Wide-angle camera over an entrance | Counted by hand once per second (46 checks; 4 ambiguous seconds left out) | Headcount **97%**: exact 98% |

**Why the camera is at the front, and why two counts.** A camera above the windscreen sees every seated passenger's face and everyone boarding at the front door. With a full bus, though, the back rows are hidden behind the heads in front, so a single-frame headcount undercounts (61% on the simulation). The app therefore uses the method real automatic passenger counters use: it tracks people across a **door line** (boarding +1, alighting −1), where everyone passes close to the camera. That count is 98% accurate on the simulation. The cabin headcount is still shown as a cross-check.

The simulated cabin is rendered by `scripts/make_simulation.py` using real people cut out (YOLOv8-seg) of the Intel IoT DevKit sample videos (Apache-2.0; cut-outs in `data/sim_cutouts`). Re-render and re-score:

```bash
python scripts/make_simulation.py --cutouts data/sim_cutouts --out frontend/sim
python -c "from backend import simulation; simulation.evaluate('bus_cabin_sim')"
```

**Test a real bus video:** log in as operator/admin, open Camera simulation, and use *Test your own bus video*. Choose where the camera is (front of the cabin, or above the door). The clip is evaluated in the background. Pause at a few moments, type the real number of people, and *Save check*: the accuracy table updates with your hand checks.

## Bus timings — how the times are built

- **Timetable** trips come from `departures_from_origin` / `departures_from_destination` in the dataset.
- **Frequency service**: a bus every 10 min at peak, 15 min off-peak, 05:30–23:00. Each route's slots are staggered so routes don't all leave on the same minute, and a frequency slot within 5 min of a timetable trip is dropped. Filter with *Timetable* / *10/15-min service*.
- **Traffic adjustment**: scheduled running time between stops is multiplied by a factor for that hour, learned from `travel_time_between_stops` in the dataset (e.g. ×0.79 at 12:00, ×1.17 at 21:00). The board shows the expected time, the scheduled time struck out, and the delay.
- Buses with a live camera counter on the route show their real on-board count.
- These are estimates built from the dataset, not BMTC's live GPS feed.

## Route start/end verification

The busiest terminals (Kempegowda Bus Station, KR Market, Shivajinagar, Banashankari TTMC, Kengeri TTMC, Central Silk Board, Hampinagara, Mysuru Road Satellite, Hebbal, Domlur, Nelamangala, Yelahanka) were checked against published coordinates (Wikipedia / OpenStreetMap / latlong.net — see `data/terminal_reference.csv`). The dataset's coordinates for these were accurate (10–400 m), and **1,786 routes** have at least one end verified. Fixes applied on import:

- **54** name variants of the same place merged (e.g. `BANASHANKARI TTMC` / `Banashankari Bus Station` → *Banashankari TTMC*; `K R Puram` / `KR Puram`), plus **83** route ends relabelled to the verified terminal's name.
- Route **369M**: labelled "Kempegowda Bus Station" but its stops start at Vijayanagara TTMC, 4.3 km away → label corrected.
- Route **306J**: ends at "Domlur", which is actually Dommaluru village near Devanagundi (25 km from Domlur TTMC, consistent with the route's own stops) → relabelled *Domlur (Dommaluru)* rather than moved.
- Routes **266**, **317Y**, **317YE**: terminal coordinates far from the route's own next stop → re-pointed to the location the other routes use.

## Using the passenger counter

1. Log in as an operator → **Passenger counter**.
2. Choose the bus and a source: **Webcam**, **Video file** or **Photo**.
3. Press **Start counting**. Position the yellow line across the doorway and choose which crossing direction counts as IN.
4. Set the **current stop** as the bus moves. Use **+1 / −1** to correct by hand.

Detector: YOLOv8n (`models/yolov8n.pt`, bundled, ~8 frames/s on a laptop CPU). If `ultralytics`/PyTorch isn't installed, it falls back to OpenCV's HOG detector.

> Webcams only work on `http://localhost` or HTTPS.

## How the dataset is used

- `ALL_ROUTES_STOP_LEVEL.csv` → routes, stops, route–stop sequences, timetables.
- Batch workbooks (`Dataset` sheet) → `trip_observations` (traffic, peak flag, occupancy, delay) → crowd model, traffic factors for bus timings, observed averages.
- Occupancy % is read against 40 seats, so the seat map shows `round(% × 40 / 100)` seats taken.
- `delay_min` in the dataset is essentially random (R² ≈ 0), so it isn't used for timings; travel times are used instead.

## Project structure

```
backend/
  app.py          FastAPI app: all APIs + static frontend
  db.py           SQLite schema, migrations
  auth.py         password hashing, JWT cookies, role guards
  ml.py           crowd forecasting (integer %, seats)
  schedule.py     bus timings: timetable + 10/15-min service, traffic-adjusted ETAs
  vision.py       person detector, tiled headcount, tracker, door-line counter
  demo.py         live headcount demo: photo/video detection, server-side video fallback
  simulation.py   clip evaluation against ground truth, uploads, hand checks
scripts/
  import_dataset.py   fix_terminals.py   train_model.py   make_simulation.py
frontend/
  index.html timings.html route.html simulation.html demo.html counter.html admin.html alerts.html login.html register.html
  js/seatmap.js   40-seat bus map     js/emergency.js  bottom-left emergency button
  demo/           sample photos & video for the Live demo page
  sim/            simulation & real test clips, ground truth (.json), evaluation results (.results.json)
data/  atobs.db  terminal_reference.csv  terminal_audit.csv  sim_cutouts/
models/ crowd_model.joblib  yolov8n.pt
```

API docs: http://localhost:8000/docs
