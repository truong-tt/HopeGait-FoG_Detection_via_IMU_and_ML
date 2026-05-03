"""HopeGait synthetic-data agent — Gemini-orchestrated FoG dataset generator.

Uses Gemini to parameterize a deterministic signal synthesizer (`synth_signal.py`),
producing `.npy` training data in the same layout `preprocess.py` writes — drop-in
for `dataset.py`. Five paced calls stay inside the free tier on
`gemini-3-flash-preview` (with `gemini-3.1-flash-lite-preview` as a fallback);
the LLM never produces raw waveforms. See `agent/README.md`.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dateutil import tz
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types as genai_types
except ImportError:
    print(
        "[ERROR] google-genai is not installed. "
        "Run: pip install -r agent/requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Local-only deps (numpy/scipy already in agent/requirements.txt).
from data_pipeline.dsp import IMUFilter  # noqa: E402

try:
    from . import synth_signal  # noqa: E402
except ImportError:
    sys.path.insert(0, str(ROOT / "agent"))
    import synth_signal  # type: ignore[no-redef]  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMU_CHANNELS: tuple[str, ...] = ("ax", "ay", "az", "gx", "gy", "gz")
FREQ_SAMPLED_HZ: int = 128
FREQ_DESIRED_HZ: int = 64
WINDOW_SAMPLES: int = 128            # 2 s @ 64 Hz — matches src/config.py
WINDOW_OVERLAP: float = 0.5
N_SUBJECTS: int = 20                 # ≈ 2× Stanford NMBL FoG-positive subjects
SUBJECT_DURATION_S: float = 300.0    # 5 min per subject — keeps prompts under TPM budget

PRIMARY_MODEL: str = "gemini-3-flash-preview"
FALLBACK_MODEL: str = "gemini-3.1-flash-lite-preview"
MAX_OUTPUT_TOKENS: int = 8192        # large output → more value per request
MIN_CALL_INTERVAL_S: float = 12.5    # ≤ 5 RPM with safety margin

DATA_DIR: Path = ROOT / "data"
SYNTH_OUT_DIR: Path = DATA_DIR / "synthetic" / f"win_{WINDOW_SAMPLES}"
REPORTS_DIR: Path = ROOT / "reports"

console = Console(force_terminal=True, width=120)


# ---------------------------------------------------------------------------
# Run-state bookkeeping
# ---------------------------------------------------------------------------
@dataclass
class GeminiCallResult:
    """One Gemini round-trip, including token accounting and any error string."""
    label: str
    model_used: str
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    error: str | None = None


@dataclass
class RunState:
    """Cumulative agent state — token totals, call history, wall-clock."""
    started_at: float = field(default_factory=time.time)
    last_call_at: float = 0.0
    gemini_results: list[GeminiCallResult] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def now_iso() -> str:
    """Local-tz wall-clock timestamp for log lines and report headers."""
    return datetime.now(timezone.utc).astimezone(tz.tzlocal()).strftime(
        "%Y-%m-%d %H:%M:%S %Z"
    )


def log_step(step_num: int, title: str) -> None:
    """Render a Rich panel banner separating each pipeline step."""
    console.print()
    console.print(
        Panel(
            f"[bold cyan]STEP {step_num}[/bold cyan] — [bold]{title}[/bold]\n"
            f"[dim]{now_iso()}[/dim]",
            border_style="cyan",
            expand=False,
        )
    )


def log_info(msg: str) -> None:
    console.print(f"[bold blue]i[/bold blue] {msg}")


def log_ok(msg: str) -> None:
    console.print(f"[bold green]+[/bold green] {msg}")


def log_warn(msg: str) -> None:
    console.print(f"[bold yellow]![/bold yellow] {msg}")


def log_err(msg: str) -> None:
    console.print(f"[bold red]x[/bold red] {msg}")


# ---------------------------------------------------------------------------
# Gemini client + paced call wrapper (free-tier-aware)
# ---------------------------------------------------------------------------
def make_genai_client() -> genai.Client:
    """Build a Gemini client from `GOOGLE_API_KEY` or exit with guidance."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        log_err("GOOGLE_API_KEY environment variable is not set.")
        log_info("Get a free key at: https://aistudio.google.com/apikey")
        sys.exit(1)
    return genai.Client(api_key=api_key)


def _pace(state: RunState) -> None:
    """Sleep until MIN_CALL_INTERVAL_S has elapsed since the last call.

    Keeps the run inside the free-tier 5 RPM cap without server-side 429s.
    """
    if state.last_call_at == 0:
        return
    wait = MIN_CALL_INTERVAL_S - (time.time() - state.last_call_at)
    if wait > 0:
        log_info(f"  pacing for free tier: sleeping {wait:.1f}s (≤ 5 RPM)")
        time.sleep(wait)


def gemini_call(
    client: genai.Client, prompt: str, label: str, state: RunState
) -> GeminiCallResult:
    """Call Gemini: gemini-3-flash-preview × 2 → gemini-3.1-flash-lite-preview fallback, with free-tier pacing."""
    _pace(state)
    log_info(f"-> Gemini call: [yellow]{label}[/yellow]")
    log_info(f"  prompt size: {len(prompt):,} chars (~{len(prompt) // 4:,} tokens)")

    cfg = genai_types.GenerateContentConfig(
        max_output_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.7,
    )
    last_err: str | None = None
    t0 = time.time()
    for attempt in (1, 2):
        try:
            resp = client.models.generate_content(
                model=PRIMARY_MODEL, contents=prompt, config=cfg
            )
            usage = getattr(resp, "usage_metadata", None)
            pt = int(getattr(usage, "prompt_token_count", 0) or 0)
            ct = int(getattr(usage, "candidates_token_count", 0) or 0)
            tt = int(getattr(usage, "total_token_count", pt + ct) or (pt + ct))
            text = getattr(resp, "text", "") or ""
            elapsed = time.time() - t0
            log_ok(
                f"  {PRIMARY_MODEL} attempt {attempt} OK — "
                f"prompt={pt:,} completion={ct:,} total={tt:,} latency={elapsed:.2f}s"
            )
            state.last_call_at = time.time()
            return GeminiCallResult(label, PRIMARY_MODEL, text, pt, ct, tt, elapsed)
        except genai_errors.APIError as exc:
            last_err = f"APIError: {exc}"
            log_warn(f"  {PRIMARY_MODEL} attempt {attempt} failed: {exc} (sleeping 2s)")
            time.sleep(2)
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            log_warn(f"  {PRIMARY_MODEL} attempt {attempt} unexpected error: {exc}")
            time.sleep(2)

    log_warn(f"  Falling back to {FALLBACK_MODEL}.")
    try:
        resp = client.models.generate_content(
            model=FALLBACK_MODEL, contents=prompt, config=cfg
        )
        usage = getattr(resp, "usage_metadata", None)
        pt = int(getattr(usage, "prompt_token_count", 0) or 0)
        ct = int(getattr(usage, "candidates_token_count", 0) or 0)
        tt = int(getattr(usage, "total_token_count", pt + ct) or (pt + ct))
        text = getattr(resp, "text", "") or ""
        elapsed = time.time() - t0
        log_ok(
            f"  {FALLBACK_MODEL} OK — prompt={pt:,} completion={ct:,} "
            f"total={tt:,} latency={elapsed:.2f}s"
        )
        state.last_call_at = time.time()
        return GeminiCallResult(label, FALLBACK_MODEL, text, pt, ct, tt, elapsed)
    except Exception as exc:
        elapsed = time.time() - t0
        err = f"{last_err} | fallback failed: {type(exc).__name__}: {exc}"
        log_err(f"  Both primary and fallback failed: {err}")
        return GeminiCallResult(label, "none", f"[ERROR] {err}", 0, 0, 0, elapsed, err)


def update_totals(state: RunState, result: GeminiCallResult) -> None:
    """Append a Gemini call result and roll its tokens into the run totals."""
    state.gemini_results.append(result)
    state.total_prompt_tokens += result.prompt_tokens
    state.total_completion_tokens += result.completion_tokens
    state.total_tokens += result.total_tokens


# ---------------------------------------------------------------------------
# JSON extraction — Gemini sometimes wraps JSON in fences or prose
# ---------------------------------------------------------------------------
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str, fallback: Any) -> Any:
    """Extract JSON from a Gemini response — fenced block, bare object, or embedded prose.

    Returns `fallback` on parse failure so a bad call doesn't abort the run.
    """
    if not text:
        return fallback
    m = _JSON_FENCE_RE.search(text)
    if m:
        candidate = m.group(1)
    else:
        # Look for the first [ or { and balance it.
        idx_list = [i for i in (text.find("["), text.find("{")) if i >= 0]
        if not idx_list:
            return fallback
        start = min(idx_list)
        candidate = text[start:]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Try the largest balanced [...] / {...} substring as a last resort.
        for open_c, close_c in (("[", "]"), ("{", "}")):
            i = candidate.find(open_c)
            if i < 0:
                continue
            depth = 0
            for j in range(i, len(candidate)):
                if candidate[j] == open_c:
                    depth += 1
                elif candidate[j] == close_c:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(candidate[i:j + 1])
                        except json.JSONDecodeError:
                            break
        log_warn("  could not parse JSON from response — using fallback.")
        return fallback


# ---------------------------------------------------------------------------
# Project + dataset context blob (padded into every prompt)
# ---------------------------------------------------------------------------
def project_context_blob() -> str:
    """Reusable context fragment included in every Gemini prompt for grounding."""
    return (
        "## HopeGait Project Context\n"
        "HopeGait is an edge-deployed Tiny-ML system for Freezing-of-Gait "
        "(FoG) detection in Parkinson's patients, intended for a soft "
        "robotic garment. The model is a strictly causal Temporal "
        "Convolutional Network (HopeGaitTCN) with channel progression "
        "(32, 64, 96, 128), kernel size 3, exponential dilation 1/2/4/8 "
        "(receptive field 61 samples ≈ 0.95 s at 64 Hz), TimeWiseLayerNorm "
        "(per-timestep), and a CausalSqueezeExcite1d block using a "
        "cumulative-mean across time so the dense head stays causal. The "
        "model has two heads: (a) last-step head for real-time MCU "
        "inference, (b) dense per-timestep auxiliary head for training "
        "supervision. Loss is focal loss with gamma=2.0 to handle "
        "FoG/normal class imbalance.\n\n"
        "## Sensor Spec (matches stanfordnmbl/imu-fog-detection upstream)\n"
        f"- IMU channels (6): {', '.join(IMU_CHANNELS)} "
        "(accelerometer ax/ay/az in m/s^2, gyroscope gx/gy/gz in rad/s)\n"
        f"- Raw sample rate (FREQ_SAMPLED): {FREQ_SAMPLED_HZ} Hz\n"
        f"- Resampled rate (FREQ_DESIRED): {FREQ_DESIRED_HZ} Hz\n"
        f"- Window length: {WINDOW_SAMPLES} samples (2 s @ {FREQ_DESIRED_HZ} Hz), "
        f"overlap {WINDOW_OVERLAP:.0%}\n"
        "- Preprocessing splits 3 accel channels into 'linear_acc' (motion-only) and "
        "'gravity' (slow component) via a 0.3 Hz Butterworth lowpass — TCN sees 9 channels.\n\n"
        "## FoG Biomarker Literature Anchors\n"
        "- Moore, Bachlin et al. (2008) Freeze Index: power(3-8 Hz) / power(0.5-3 Hz) on "
        "vertical-component linear acceleration.\n"
        "- Typical gait stride frequency: 0.8-1.3 Hz; step frequency: 1.5-2.5 Hz.\n"
        "- FoG episode durations cluster bimodally: trembling-in-place (1-3 s) and "
        "akinesia (5-15 s, occasionally >30 s).\n"
        "- NFOG-Q (New Freezing of Gait Questionnaire) scores 0-28; clinical FoG-positive "
        "patients typically score 8-24.\n"
        "- Hoehn & Yahr stages 1-5 — FoG most commonly seen in stages 2.5-4.\n\n"
        "## Synthetic-Data Goal\n"
        f"Generate {N_SUBJECTS} synthetic subjects, {SUBJECT_DURATION_S:.0f} s each, "
        "with realistic distributions over age, sex, NFOG-Q severity, gait frequency, "
        "tremor band, and FoG event timing. The numerical waveform synthesis is "
        "deterministic Python (gait oscillator + freeze-band injection + tremor + sensor "
        "noise) — your job is to *parameterize* it with clinically plausible values, not "
        "to produce raw waveforms.\n"
    )


# ---------------------------------------------------------------------------
# Prompt builders for the 5 calls
# ---------------------------------------------------------------------------
def build_profiles_prompt() -> str:
    """Call 1 — design N synthetic clinical subjects (JSON array)."""
    return (
        project_context_blob()
        + "\n## Task — Subject Profile Design\n"
        f"Return a JSON array of exactly {N_SUBJECTS} synthetic Parkinson's subjects "
        "as objects with these fields:\n\n"
        "- `subject_id` (int, 1..N)\n"
        "- `age` (int, 50-85)\n"
        "- `sex` ('M' | 'F')\n"
        "- `years_with_pd` (int, 1-25)\n"
        "- `hoehn_yahr_stage` (float, 1.0/1.5/2.0/2.5/3.0/3.5/4.0)\n"
        "- `nfog_q` (int, 0-28; sample mostly from clinically FoG-positive range)\n"
        "- `gait_freq_hz` (float, 0.8-2.5 — slower for advanced PD)\n"
        "- `tremor_band_hz` ([float, float] within 3.5-7.5)\n"
        "- `tremor_amp_g` (float, 0.0-0.30 fraction of g; correlate with H&Y)\n"
        "- `fog_severity` (int, derived from nfog_q; 0-28)\n"
        "- `fog_episode_rate_per_min` (float, 0.05-0.6)\n"
        "- `fog_mean_duration_s` (float, 1.5-15.0)\n"
        "- `fog_duration_std_s` (float, 0.5-6.0)\n"
        "- `notes` (short clinical vignette, 1-2 sentences)\n\n"
        "Sample so that ~70% of subjects are clinically FoG-positive (NFOG-Q >= 8) and "
        "~30% are non-FoG controls (NFOG-Q < 4 — set fog_episode_rate_per_min near 0). "
        "Make the cohort heterogeneous: vary age, sex, severity, gait speed, tremor band. "
        "Distinct subjects, not duplicates with id changes.\n\n"
        "Return ONLY a JSON array — no surrounding prose, no markdown fence."
    )


def build_timelines_prompt(profiles: list[dict[str, Any]]) -> str:
    """Call 2 — per-subject FoG event timelines (JSON object keyed by subject_id)."""
    return (
        project_context_blob()
        + "\n## Subject Profiles (from prior call)\n```json\n"
        + json.dumps(profiles, indent=2)
        + "\n```\n\n## Task — Event Timeline Design\n"
        f"For each subject, generate a list of timed events covering the full "
        f"{SUBJECT_DURATION_S:.0f}-second recording. Each event is an object with:\n\n"
        "- `start_s` (float, seconds from recording start)\n"
        "- `duration_s` (float, seconds)\n"
        "- `type` ('walk' | 'fog' | 'transition')\n"
        "- `intensity` (float 0.0-1.0; for FoG, scales freeze-band amplitude)\n"
        "- `note` (very short clinical descriptor, e.g. 'turning trigger', 'doorway')\n\n"
        "Constraints:\n"
        f"- The events MUST cover [0, {SUBJECT_DURATION_S:.0f}] s with no gaps.\n"
        "- A subject's FoG episode count should match their `fog_episode_rate_per_min` "
        f"× ({SUBJECT_DURATION_S/60:.1f} min) within ±1.\n"
        "- FoG episode durations should be sampled from N(fog_mean_duration_s, "
        "fog_duration_std_s) and clipped to [0.8, 30] s.\n"
        "- Insert short 0.5-2 s 'transition' segments around FoG to mimic gait hesitation.\n"
        "- Non-FoG controls (nfog_q < 4) should have zero or one tiny FoG episode.\n"
        "- Avoid more than one FoG every 6 s on average; cluster a few near 'turning' or "
        "'doorway' triggers as the literature suggests.\n\n"
        "Return ONLY a JSON object of shape `{ \"<subject_id>\": [event, event, ...], ... }` — "
        "no prose, no markdown fence."
    )


def build_qc_prompt(
    profiles: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    prior_calls: list[GeminiCallResult],
) -> str:
    """Call 3 — review per-subject signal stats and propose parameter corrections."""
    chained = "\n\n".join(
        f"### Prior call: {r.label} ({r.total_tokens:,} tokens)\n{r.text[:4000]}"
        for r in prior_calls
    )
    return (
        project_context_blob()
        + "\n## Profiles\n```json\n" + json.dumps(profiles, indent=2) + "\n```\n"
        + "\n## Per-subject Synthesis Stats\n```json\n"
        + json.dumps(summaries, indent=2) + "\n```\n"
        + "\n## Chained Prior Analyses\n\n" + chained + "\n\n"
        "## Task — Synthesis QC\n"
        "Each subject above was synthesized once with the parameters from call 1. The "
        "stats include `walk_freeze_index`, `fog_freeze_index`, channel mean/std, and "
        "FoG fraction. For a clinically realistic recording we expect:\n\n"
        "- `walk_freeze_index` typically 0.05-0.4 (low — gait energy concentrated in 0.5-3 Hz)\n"
        "- `fog_freeze_index` typically 1.5-6.0 (high — freeze-band power exceeds locomotor)\n"
        "- accelerometer std mostly 0.5-3.0 m/s^2 across walking subjects; az std lower\n"
        "  because the gravity baseline dominates\n"
        "- gyroscope std mostly 0.1-1.5 rad/s\n\n"
        "Identify subjects whose stats are clinically off and return JSON corrections of shape:\n\n"
        "```\n"
        "{ \"<subject_id>\": { \"gait_freq_hz\": <new>, \"tremor_amp_g\": <new>, "
        "\"fog_severity\": <new>, \"reason\": \"<short>\" }, ... }\n"
        "```\n\n"
        "Only include subjects that need adjustment (omit ones that look fine). Return "
        "ONLY the JSON object, no prose, no fence."
    )


def build_augmentation_prompt(
    profiles: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    prior_calls: list[GeminiCallResult],
) -> str:
    """Call 4 — recommend augmentation strategy parameters."""
    chained = "\n\n".join(
        f"### Prior call: {r.label}\n{r.text[:3500]}" for r in prior_calls
    )
    return (
        project_context_blob()
        + "\n## Final per-subject stats\n```json\n"
        + json.dumps(summaries, indent=2) + "\n```\n"
        + "\n## Chained Prior Analyses\n\n" + chained + "\n\n"
        "## Task — Augmentation Strategy\n"
        "Recommend a training-time augmentation policy. The repo currently uses joint "
        "SO(3) rotation augmentation on (linear_acc, gravity, gyro) with rotation_max_deg "
        "in config and time-shift via roll. Suggest concrete values for the synthetic "
        "subset specifically (it can stand more aggressive augmentation than real data) "
        "as a JSON object:\n\n"
        "```\n"
        "{ \"rotation_max_deg\": <float>, \"rotation_prob\": <float>, \n"
        "  \"time_shift_max_samples\": <int>, \"time_shift_prob\": <float>,\n"
        "  \"gaussian_noise_std\": <float>, \"channel_dropout_prob\": <float>,\n"
        "  \"rationale\": \"<2-4 sentences>\" }\n"
        "```\n\n"
        "Followed by a 200-400-word free-text rationale section that cites the specific "
        "stats above. Return JSON first, then the prose."
    )


def build_dataset_card_prompt(
    profiles: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    prior_calls: list[GeminiCallResult],
) -> str:
    """Call 5 — write the dataset card and propose LOSO splits."""
    chained = "\n\n".join(
        f"### Prior call: {r.label}\n{r.text[:3000]}" for r in prior_calls
    )
    return (
        project_context_blob()
        + "\n## Profiles\n```json\n" + json.dumps(profiles, indent=2) + "\n```\n"
        + "\n## Final per-subject stats\n```json\n"
        + json.dumps(summaries, indent=2) + "\n```\n"
        + "\n## Chained Prior Analyses\n\n" + chained + "\n\n"
        "## Task — Dataset Card + LOSO Split\n"
        "Write a complete Markdown dataset card following the Hugging Face / Gebru "
        "(Datasheets-for-Datasets) structure. Include:\n\n"
        "1. **Dataset name and version** (HopeGait-Synthetic v0.1)\n"
        "2. **Intended use** (research-only synthetic supplement to "
        "stanfordnmbl/imu-fog-detection; never to be used as the sole training source for "
        "any clinical claim)\n"
        "3. **Generation method** (LLM-parameterized deterministic gait + freeze-band + "
        "tremor synthesis at 64 Hz; cite our pipeline)\n"
        "4. **Composition** (per-subject stats summary table)\n"
        "5. **Limitations** (no real inter-subject variability, no real sensor mounting "
        "drift, no comorbid medical conditions, simplified episode structure)\n"
        "6. **Ethical considerations** (synthetic, no PHI, but downstream model trained on "
        "this must not be marketed as clinically validated)\n"
        "7. **Recommended usage** (mix-in proportion with real data, e.g. 1:1 or 1:2)\n"
        "8. **Proposed LOSO split** — assign each subject_id to 'train' / 'val' / 'test' so "
        "that severity, sex, and age are balanced across splits.\n\n"
        "End with a JSON block:\n\n"
        "```\n"
        "{\"loso_split\": {\"<subject_id>\": \"train|val|test\", ...}}\n"
        "```\n\n"
        "Aim for at least 800 words of prose before the JSON block."
    )


# ---------------------------------------------------------------------------
# Local synthesis + windowing
# ---------------------------------------------------------------------------
def synthesize_all_subjects(
    profiles: list[dict[str, Any]], timelines: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Render all subjects' raw 6-channel signals + sample-level labels."""
    out: list[dict[str, Any]] = []
    for prof in profiles:
        sid = str(prof.get("subject_id"))
        events = timelines.get(sid, []) or timelines.get(int(sid), []) if sid.isdigit() else []
        if not events:
            log_warn(f"  no timeline for subject {sid} — defaulting to all-walk.")
            events = [{"start_s": 0.0, "duration_s": SUBJECT_DURATION_S, "type": "walk"}]
        signal, labels = synth_signal.synthesize_subject(
            prof, events, SUBJECT_DURATION_S, fs=FREQ_DESIRED_HZ
        )
        out.append({"profile": prof, "events": events, "signal": signal, "labels": labels})
    return out


def apply_corrections(
    subjects: list[dict[str, Any]], corrections: dict[str, dict[str, Any]]
) -> int:
    """Apply Gemini-proposed parameter deltas in-place and re-synthesize touched subjects."""
    n_changed = 0
    for s in subjects:
        sid = str(s["profile"].get("subject_id"))
        corr = corrections.get(sid) or corrections.get(int(sid)) if sid.isdigit() else None
        if not corr:
            continue
        for k, v in corr.items():
            if k == "reason":
                continue
            if k in s["profile"]:
                s["profile"][k] = v
        s["signal"], s["labels"] = synth_signal.synthesize_subject(
            s["profile"], s["events"], SUBJECT_DURATION_S, fs=FREQ_DESIRED_HZ
        )
        n_changed += 1
    return n_changed


def expand_to_9_channels(signal_6: np.ndarray, fs: float) -> np.ndarray:
    """6 raw channels → 9 model channels via IMUFilter — matches preprocess.py bit-for-bit."""
    imu = IMUFilter(fs=fs)
    acc = signal_6[:, :3]
    gyro = signal_6[:, 3:]
    linear_acc, gravity, gyro_f, _ = imu.process_signal(acc, gyro, timestamps=None)
    return np.hstack([linear_acc, gravity, gyro_f]).astype(np.float32)


def window_and_save(
    subjects: list[dict[str, Any]], out_dir: Path, fs: float
) -> dict[str, Any]:
    """DSP split + windowing + save .npy files matching preprocess.py layout.

    Writes subj_S<id>_synth_x.npy (N, T, 9) float32 and _y.npy (N, T) int64;
    dataset.py LOSO loader picks them up alongside real recordings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, int(WINDOW_SAMPLES * (1 - WINDOW_OVERLAP)))
    manifest: dict[str, Any] = {"subjects": [], "total_windows": 0,
                                 "total_fog_windows": 0, "out_dir": str(out_dir)}
    for s in subjects:
        sid = s["profile"]["subject_id"]
        signal_9 = expand_to_9_channels(s["signal"], fs)
        labels = s["labels"]
        n = min(len(signal_9), len(labels))
        x_windows: list[np.ndarray] = []
        y_windows: list[np.ndarray] = []
        for i in range(0, n - WINDOW_SAMPLES + 1, step):
            x_windows.append(signal_9[i:i + WINDOW_SAMPLES])
            y_windows.append(labels[i:i + WINDOW_SAMPLES])
        if not x_windows:
            continue
        x_np = np.stack(x_windows).astype(np.float32)
        y_np = np.stack(y_windows).astype(np.int64)
        np.save(out_dir / f"subj_S{sid}_synth_x.npy", x_np)
        np.save(out_dir / f"subj_S{sid}_synth_y.npy", y_np)
        n_fog = int((y_np[:, -1] == 1).sum())
        manifest["subjects"].append({
            "subject_id": sid,
            "n_windows": int(x_np.shape[0]),
            "n_fog_windows": n_fog,
            "fog_rate_last_step": float(n_fog / max(x_np.shape[0], 1)),
        })
        manifest["total_windows"] += int(x_np.shape[0])
        manifest["total_fog_windows"] += n_fog
    return manifest


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_token_panel(state: RunState) -> Panel:
    """Final summary panel — per-call token usage plus totals."""
    table = Table(show_header=True, header_style="bold magenta")
    for col in ("call", "model", "prompt", "completion", "total", "latency_s"):
        table.add_column(col)
    for r in state.gemini_results:
        table.add_row(
            r.label, r.model_used,
            f"{r.prompt_tokens:,}", f"{r.completion_tokens:,}",
            f"{r.total_tokens:,}", f"{r.latency_s:.2f}",
        )
    table.add_row(
        "[bold]TOTAL[/bold]", "—",
        f"[bold]{state.total_prompt_tokens:,}[/bold]",
        f"[bold]{state.total_completion_tokens:,}[/bold]",
        f"[bold]{state.total_tokens:,}[/bold]",
        f"[bold]{(time.time() - state.started_at):.2f}[/bold]",
    )
    return Panel(table, title="[bold]=== AGENT RUN COMPLETE ===[/bold]",
                 border_style="green", expand=False)


def save_run_report(
    state: RunState,
    profiles: list[dict[str, Any]],
    summaries_initial: list[dict[str, Any]],
    summaries_final: list[dict[str, Any]],
    save_manifest: dict[str, Any],
    sections: list[tuple[str, str]],
) -> Path:
    """Write a timestamped Markdown report bundling profiles, stats, and Gemini outputs."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz.tzlocal()).strftime("%Y-%m-%d_%H-%M-%S")
    out = REPORTS_DIR / f"{stamp}_synth_run.md"
    runtime_s = time.time() - state.started_at

    parts: list[str] = []
    parts.append(f"# HopeGait Synthetic-Data Agent Run — {stamp}\n\n")
    parts.append(f"_Generated at {now_iso()}_  \n")
    parts.append(f"**Runtime:** {runtime_s:.2f} s  \n")
    parts.append(
        f"**Tokens:** prompt={state.total_prompt_tokens:,}, "
        f"completion={state.total_completion_tokens:,}, "
        f"total={state.total_tokens:,}\n\n"
    )
    parts.append("## Output\n\n")
    parts.append("```json\n" + json.dumps(save_manifest, indent=2) + "\n```\n\n")
    parts.append("## Per-call Token Usage\n\n")
    parts.append("| call | model | prompt | completion | total | latency_s |\n")
    parts.append("|---|---|---|---|---|---|\n")
    for r in state.gemini_results:
        parts.append(
            f"| {r.label} | {r.model_used} | {r.prompt_tokens:,} | "
            f"{r.completion_tokens:,} | {r.total_tokens:,} | {r.latency_s:.2f} |\n"
        )
    parts.append("\n## Subject Profiles\n\n```json\n")
    parts.append(json.dumps(profiles, indent=2))
    parts.append("\n```\n\n## Initial QC Stats\n\n```json\n")
    parts.append(json.dumps(summaries_initial, indent=2))
    parts.append("\n```\n\n## Final QC Stats (after corrections)\n\n```json\n")
    parts.append(json.dumps(summaries_final, indent=2))
    parts.append("\n```\n")
    for title, body in sections:
        parts.append(f"\n## {title}\n\n{body}\n")
    out.write_text("".join(parts), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    state = RunState()
    console.print(
        Panel(
            "[bold cyan]HopeGait Synthetic-Data Agent[/bold cyan]\n"
            "[dim]Gemini-orchestrated synthetic FoG dataset generator\n"
            f"Started {now_iso()}[/dim]",
            border_style="bright_cyan",
        )
    )
    client = make_genai_client()

    # ---- STEP 1 — Profile design (Gemini #1) ----
    log_step(1, "Subject profile design (Gemini #1)")
    r1 = gemini_call(client, build_profiles_prompt(), "subject_profiles", state)
    update_totals(state, r1)
    profiles = extract_json(r1.text, fallback=[])
    if not isinstance(profiles, list) or not profiles:
        log_err("Could not extract subject profiles — aborting.")
        return 1
    log_ok(f"Got {len(profiles)} subject profiles.")
    if r1.text:
        console.print(Panel(Markdown(r1.text[:3000]),
                            title="Gemini #1 — Profiles (truncated)",
                            border_style="blue"))

    # ---- STEP 2 — Event timelines (Gemini #2) ----
    log_step(2, "Per-subject FoG event timelines (Gemini #2)")
    r2 = gemini_call(client, build_timelines_prompt(profiles), "event_timelines", state)
    update_totals(state, r2)
    timelines_obj = extract_json(r2.text, fallback={})
    if not isinstance(timelines_obj, dict):
        log_warn("Timelines came back non-dict — defaulting to all-walk recordings.")
        timelines_obj = {}
    log_ok(f"Got timelines for {len(timelines_obj)} subjects.")
    if r2.text:
        console.print(Panel(Markdown(r2.text[:3000]),
                            title="Gemini #2 — Timelines (truncated)",
                            border_style="blue"))

    # ---- STEP 3 — Local synthesis ----
    log_step(3, "Local IMU signal synthesis (deterministic numpy)")
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:
        task = progress.add_task("Synthesizing subjects", total=len(profiles))
        subjects = []
        for prof in profiles:
            sid = str(prof.get("subject_id"))
            events = timelines_obj.get(sid) or timelines_obj.get(int(sid)) \
                if sid.isdigit() else timelines_obj.get(sid)
            if not events:
                events = [{"start_s": 0.0, "duration_s": SUBJECT_DURATION_S, "type": "walk"}]
            signal, labels = synth_signal.synthesize_subject(
                prof, events, SUBJECT_DURATION_S, fs=FREQ_DESIRED_HZ
            )
            subjects.append({"profile": prof, "events": events,
                             "signal": signal, "labels": labels})
            progress.advance(task)

    summaries_initial = [synth_signal.per_subject_summary(s["signal"], s["labels"],
                                                          fs=FREQ_DESIRED_HZ)
                         for s in subjects]
    log_ok(f"Synthesized {len(subjects)} subjects, "
           f"{sum(s['signal'].shape[0] for s in subjects):,} total samples.")

    # ---- STEP 4 — QC critique (Gemini #3) ----
    log_step(4, "Synthesis QC critique (Gemini #3)")
    r3 = gemini_call(client, build_qc_prompt(profiles, summaries_initial, [r1, r2]),
                     "qc_critique", state)
    update_totals(state, r3)
    corrections = extract_json(r3.text, fallback={})
    if not isinstance(corrections, dict):
        corrections = {}
    log_ok(f"Got corrections for {len(corrections)} subjects.")
    if r3.text:
        console.print(Panel(Markdown(r3.text[:3000]),
                            title="Gemini #3 — QC Critique (truncated)",
                            border_style="blue"))

    # ---- STEP 5 — Re-synthesize corrected subjects ----
    log_step(5, "Apply parameter corrections + regenerate")
    n_changed = apply_corrections(subjects, corrections)
    summaries_final = [synth_signal.per_subject_summary(s["signal"], s["labels"],
                                                        fs=FREQ_DESIRED_HZ)
                       for s in subjects]
    log_ok(f"Re-synthesized {n_changed} subjects with corrections applied.")

    # ---- STEP 6 — Augmentation strategy (Gemini #4) ----
    log_step(6, "Augmentation strategy recommendation (Gemini #4)")
    r4 = gemini_call(
        client,
        build_augmentation_prompt(
            [s["profile"] for s in subjects], summaries_final, [r1, r2, r3]
        ),
        "augmentation_strategy", state
    )
    update_totals(state, r4)
    if r4.text:
        console.print(Panel(Markdown(r4.text[:3000]),
                            title="Gemini #4 — Augmentation",
                            border_style="blue"))

    # ---- STEP 7 — Dataset card + LOSO split (Gemini #5) ----
    log_step(7, "Dataset card + LOSO split (Gemini #5)")
    r5 = gemini_call(
        client,
        build_dataset_card_prompt(
            [s["profile"] for s in subjects], summaries_final, [r1, r2, r3, r4]
        ),
        "dataset_card", state
    )
    update_totals(state, r5)
    if r5.text:
        console.print(Panel(Markdown(r5.text[:4000]),
                            title="Gemini #5 — Dataset Card (truncated)",
                            border_style="blue"))

    # ---- STEP 8 — DSP split + window + save ----
    log_step(8, f"DSP split + windowing + save to {SYNTH_OUT_DIR.relative_to(ROOT)}")
    save_manifest = window_and_save(subjects, SYNTH_OUT_DIR, fs=FREQ_DESIRED_HZ)
    log_ok(
        f"Wrote {save_manifest['total_windows']:,} windows "
        f"({save_manifest['total_fog_windows']:,} last-step FoG positives) "
        f"across {len(save_manifest['subjects'])} subjects."
    )
    # Save a sidecar dataset card alongside the .npy files.
    if r5.text:
        (SYNTH_OUT_DIR / "DATASET_CARD.md").write_text(r5.text, encoding="utf-8")
        log_ok(f"Wrote DATASET_CARD.md to {SYNTH_OUT_DIR.relative_to(ROOT)}")

    # ---- STEP 9 — Cumulative report ----
    log_step(9, "Cumulative report generation")
    sections = [
        ("Step 1 — Subject Profile Design", r1.text),
        ("Step 2 — Event Timelines", r2.text),
        ("Step 4 — QC Critique", r3.text),
        ("Step 6 — Augmentation Strategy", r4.text),
        ("Step 7 — Dataset Card + LOSO Split", r5.text),
    ]
    out_path = save_run_report(state, profiles, summaries_initial, summaries_final,
                                save_manifest, sections)
    log_ok(f"Report written to: {out_path.relative_to(ROOT)}")

    runtime_s = time.time() - state.started_at
    log_ok(
        f"Total tokens — prompt={state.total_prompt_tokens:,}, "
        f"completion={state.total_completion_tokens:,}, "
        f"total={state.total_tokens:,} | runtime={runtime_s:.2f}s"
    )
    console.print(render_token_panel(state))

    if any(r.error for r in state.gemini_results):
        log_warn("One or more Gemini calls degraded to fallback or failed; "
                 "see above for details.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Interrupted by user.[/bold red]")
        sys.exit(130)
    except Exception as exc:
        console.print(f"\n[bold red]Unhandled error:[/bold red] {exc}")
        traceback.print_exc()
        sys.exit(2)