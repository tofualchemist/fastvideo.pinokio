# FastVideo MLX (Pinokio launcher)

A minimal, iterative web UI for [FastVideo](https://github.com/hao-ai-lab/FastVideo)'s
Apple Silicon models, running natively through FastVideo's MLX runtime: the **FastH3**
family (distilled MiniMax H3, video with synchronized audio) and the **FastMetal** family
(distilled Wan 2.1, silent video, minutes per clip).

Supported models (all downloaded on first use, into this launcher's `models/` folder):

| Model | What it does | Download | Notes |
|---|---|---|---|
| FastMetal 1.3B | Silent video, 480p, 5 s at 16 fps, 3 steps | 1.5 GB (+12 GB shared) | About 90 s per draft on an M1 Max. Good for quick iteration. |
| FastMetal 14B | Silent video, 480p or 720p, 5 s at 16 fps, 3 steps | 15 GB (+12 GB shared) | Best Wan quality. |
| FastMetal 5B | Silent video, 480p or 720p, 5 s at 24 fps, 3 steps | 20 GB | 720p draft in about 2.5 min on an M1 Max. Own encoder and VAE. |

The FastH3 models (with audio):

| Model | What it does | Download | Notes |
|---|---|---|---|
| FastH3 Trim | 8-step pruned text to video with audio | 15 GB | Only ready-made INT6 download, so it is the default first model. Quality below V2. |
| FastH3 V1 | 4-step text to video with audio, dense attention | 70 GB | Fastest per clip. Converted to INT6 locally (17 GB kept). |
| FastH3 V2 | 8-step text to video with audio, sparse attention | 70 GB | Highest quality text model. Converted to INT6 locally (17 GB kept). |
| FastH3 Omni | Reference to video with audio: prompt + ordered image / video / voice references | 70 GB | Converted to INT6 locally (17 GB kept). 5, 10 or 15 s clips. |

Shared H3 components (NVFP4 text encoder, audio VAE, FastH3 decoders, LightVAE encoder,
about 18 GB) are downloaded once and shared by every FastH3 model. FastMetal models share a
separate 12 GB set (UMT5 text encoder, Wan VAE). The Hub repos named `*-MLX-INT6` for V1
contain only a manifest (the weights file is missing as of October 2026), which is why V1 converts locally.

## Requirements

- Apple Silicon Mac, macOS 14 or newer
- 36 GB unified memory or more for FastH3 (the lab validated on an M4 Max with 36 GB); FastMetal 1.3B and 5B run on 16 GB
- On an M1 Max, FastH3 Omni needs the default watchdog settings plus a step cooldown (Settings) and works with image
  references; video-clip references did not survive macOS's GPU watchdog on that chip
- Disk: 35 GB for the default model (Trim); V1, V2 and Omni need about 95 GB free while converting
- Expect minutes per clip, not seconds. On an M1 Max a 5 s 480p draft takes about 8 minutes with V1 and
  17 with V2 or Trim; a final roughly doubles that. The picker shows the estimate for this Mac (see
  "Time estimates" below). Draft mode is the default.

## How to use

1. **Install** from the Pinokio sidebar (clones FastVideo at a pinned commit and sets up the venv).
2. **Start**, then open the web UI.
3. Create a project, write a prompt, and **Generate**. The first run downloads the shared
   components and the selected model, with progress on the card. The model chip in the composer
   opens a picker that shows, for every model, what it makes, whether it fits this Mac, how long a
   draft and a final take here (estimated for this Mac, measured once you have run that shape), peak
   memory, and download size. It also lets you download a model ahead of time or remove one from disk.
4. Iterate from the timeline:
   - **Reroll**: same prompt and settings, new seed.
   - **Edit**: loads that turn into the composer; change anything and generate again. The seed stays
     locked so the composition stays comparable, and the new card shows a diff against its parent.
   - **Finalize**: renders that prompt and seed at full quality (all 124 frames, 36-layer decoder). A draft
     made with the default temporal shortcut denoises 73 frames, so its noise differs from the final's and the
     final is a new take of the same prompt, not an upscale of the draft. For a draft that finalizes into the
     same take, set Advanced → draft mode to "none" (full frames, fast decoder only).
   - **Use as reference** (Omni): adds the clip as a video reference for the next turn.
5. Projects live in the sidebar. A blue dot means a job is rendering there, a green dot means a new
   result is waiting.

Draft vs. final: for FastH3, drafts use the 8-layer decoder plus temporal fast mode (text models) or a
reduced reference canvas (Omni), finals use the 36-layer decoder and the full pipeline. For FastMetal,
drafts denoise half the frames and interpolate, decoding with the fast TAEHV decoder; finals denoise
every frame and decode with the full Wan VAE.

Prompts: H3 responds to structured prompts. Use `(S1)` to introduce a speaker and
`<d>[English] words</d>` for spoken lines, then describe the soundscape. FastMetal takes a plain
visual description. "Insert template" gives you a skeleton for whichever model is selected.

## Memory while rendering

FastH3 pins ("wires") GPU memory so macOS cannot page the model out mid-step. The launcher uses
FastVideo's recipe: about two thirds of unified memory (42 GiB on a 64 GB Mac, 24 on 36 GB), never more
than the working set Metal recommends for the device. Set an explicit value in Settings to override it;
"off" disables wiring.

Rendering still keeps the GPU at 100 %, and macOS shares that GPU with the display, so expect some
UI stutter while a clip renders; that part is not something the launcher can change.

## Time estimates

The time shown on the Generate button and in the picker comes from a cost model, not from a table of
chips. Each denoise step costs `s × (linear FLOPs + R × attention FLOPs)`: the two FLOP counts follow
from the model's architecture and the number of tokens in the sequence (which the launcher computes
from frames, canvas and references exactly as the runtime does), `R` is how much more an attention FLOP
costs than a matmul FLOP on this GPU, and `s` is this Mac's seconds per matmul FLOP. Before the first run
`s` comes from a 15-second GPU micro-benchmark that Install runs (`app/bench.py`; Start runs it if it is
missing). Every finished render records its token count, step times and phase times in `data/stats.json`,
and from then on `s` (and `R`, once runs at different lengths exist) are fitted from those runs. Load,
prompt encoding, decoding and frame interpolation are recorded per model and scaled by what they depend on.
Changing the clip length, canvas or references changes the token count, so the estimate follows without
any per-length factors. "measured" means this exact shape has run here; everything else says "est.".

On the M1 Max that built this launcher, the benchmark-only step estimate was within 5 % of the actual
step time for six of nine recorded FastH3 runs; the other three were early Trim drafts that ran 20 to
100 % slower than a later run of the identical shape, so the model uses medians and such runs do not
move the fit.

## Folders

- `app/` the Studio server and UI (FastAPI + a single static page, no build step, works offline)
- `FastVideo/` FastVideo checkout (created by install)
- `env/` Python environment (created by install)
- `models/` downloaded and converted weights
- `data/projects/<id>/` each project's `project.json`, `clips/` and `refs/`
- `data/logs/runner.log` the model runtime's log
- `data/stats.json` this Mac's GPU benchmark and the recorded runs the time estimates are fitted from

## API

The server is a plain HTTP JSON API. Base URL is the one shown by Pinokio, for example
`http://127.0.0.1:8798`.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/state` | models, queue, settings, projects |
| GET | `/api/models` · POST `/api/models/{id}/install` · DELETE `/api/models/{id}` | model management |
| GET/POST | `/api/projects` | list / create (`{"name": "..."}`) |
| GET/PATCH/DELETE | `/api/projects/{pid}` | project with turns / rename / delete |
| POST | `/api/projects/{pid}/uploads` | multipart `file` → reference file (`{"path", "media_type"}`) |
| POST | `/api/projects/{pid}/turns` | create a turn (see below) |
| POST | `/api/projects/{pid}/turns/{tid}/cancel` · DELETE `/api/projects/{pid}/turns/{tid}` | cancel / delete |
| GET | `/api/events` | Server-sent events: `turn`, `queue`, `projects`, `models`, `install`, `log`, `notify` |
| GET | `/files/projects/{pid}/clips/{tid}.mp4` | the rendered clip |

Turn body:

```json
{
  "prompt": "(S1) A red fox in snow says <d>[English] Hello.</d> Soundscape: wind.",
  "model": "fasth3-trim",
  "quality": "draft",
  "duration_s": 5,
  "seed": null,
  "parent_id": null,
  "refs": [{"media_type": "image", "path": "refs/u_1234_face.png"}, {"media_type": "video", "from_turn": "t_abcd"}]
}
```

### curl

```bash
BASE=http://127.0.0.1:8798
PID=$(curl -s -X POST $BASE/api/projects -H 'Content-Type: application/json' -d '{"name":"api test"}' | python3 -c 'import sys,json;print(json.load(sys.stdin)["id"])')
curl -s -X POST $BASE/api/projects/$PID/turns -H 'Content-Type: application/json' \
  -d '{"prompt":"A red fox in snow. Soundscape: wind.","model":"fasth3-trim","quality":"draft"}'
curl -s $BASE/api/projects/$PID | python3 -m json.tool
```

### Python

```python
import requests, time
base = "http://127.0.0.1:8798"
pid = requests.post(f"{base}/api/projects", json={"name": "api test"}).json()["id"]
turn = requests.post(f"{base}/api/projects/{pid}/turns",
                     json={"prompt": "A red fox in snow. Soundscape: wind.", "model": "fasth3-trim"}).json()
while True:
    t = next(t for t in requests.get(f"{base}/api/projects/{pid}").json()["turns"] if t["id"] == turn["id"])
    if t["status"] in ("done", "failed", "cancelled"):
        break
    print(t["status"], t.get("progress", {}).get("message"))
    time.sleep(5)
if t["status"] == "done":
    open("clip.mp4", "wb").write(requests.get(f"{base}/files/projects/{pid}/{t['result']['clip']}").content)
```

### JavaScript

```js
const base = "http://127.0.0.1:8798";
const project = await (await fetch(`${base}/api/projects`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: "api test" }) })).json();
const turn = await (await fetch(`${base}/api/projects/${project.id}/turns`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ prompt: "A red fox in snow. Soundscape: wind.", model: "fasth3-v1" }) })).json();
const events = new EventSource(`${base}/api/events`);
events.onmessage = (m) => { const ev = JSON.parse(m.data); if (ev.type === "turn" && ev.turn.id === turn.id) console.log(ev.turn.status, ev.turn.progress); };
```

## Development

`FASTH3_STUDIO_FAKE=1 python app/server.py` runs the same server with a fake runner that produces
test clips in seconds, for working on the UI without weights. The runner protocol
(`app/fasth3_studio/runner.py`) is the only backend-specific boundary: a CUDA runner would implement
the same two commands.

## Credits

- FastVideo and the FastH3 models: [Hao AI Lab](https://haoailab.com/FastVideo/)
- MiniMax H3 (base model) is under the MiniMax-H3 Community License; Qwen3-VL and LightVAE are Apache-2.0
