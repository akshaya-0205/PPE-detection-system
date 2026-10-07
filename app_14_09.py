import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
from pathlib import Path
import cv2
import time
import os
import json
import io
import wave
import struct
import math
import base64
import pandas as pd
from datetime import datetime

st.set_page_config(page_title="Construction Safety", layout="wide")

DANGER_ZONE_SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "danger_zone_settings.json")
DEFAULT_DANGER_BUFFER_RATIO = 0.15  # how far the danger zone extends beyond machinery, as a
                                     # fraction of the frame's average dimension

def load_danger_buffer_ratio():
    if os.path.exists(DANGER_ZONE_SETTINGS_FILE):
        try:
            with open(DANGER_ZONE_SETTINGS_FILE) as f:
                return json.load(f).get("buffer_ratio", DEFAULT_DANGER_BUFFER_RATIO)
        except Exception:
            return DEFAULT_DANGER_BUFFER_RATIO
    return DEFAULT_DANGER_BUFFER_RATIO

def save_danger_buffer_ratio(ratio):
    try:
        with open(DANGER_ZONE_SETTINGS_FILE, "w") as f:
            json.dump({"buffer_ratio": ratio}, f, indent=2)
    except Exception:
        pass

if "page" not in st.session_state:
    st.session_state.page = "home"

if "violation_logs" not in st.session_state:
    st.session_state.violation_logs = []

if "active_alerts" not in st.session_state:
    st.session_state.active_alerts = []

if "active_emergencies" not in st.session_state:
    st.session_state.active_emergencies = []

if "emergency_cooldown" not in st.session_state:
    st.session_state.emergency_cooldown = {}

if "danger_buffer_ratio" not in st.session_state:
    st.session_state.danger_buffer_ratio = load_danger_buffer_ratio()

if "sms_status" not in st.session_state:
    st.session_state.sms_status = []

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

VIOLATION_LABELS = ["NO-Hardhat", "NO-Mask", "NO-Safety Vest"]
HAZARD_LABELS = ["machinery", "vehicle"]  # dangerous equipment classes already in the model
VIOLATION_COOLDOWN_SECONDS = 8  # don't re-log the same worker's same violation more often than this
EMERGENCY_COOLDOWN_SECONDS = 20  # don't re-send an emergency alert/SMS for the same worker+zone more often than this

HAZARD_BUFFER_MULTIPLIER = {"vehicle": 1.3, "machinery": 1.0}  # vehicles move, so they get a
                                                                 # larger effective danger radius

def in_danger_zone(person_box, hazards, frame_w, frame_h, buffer_ratio):
    """A worker is 'in the danger zone' if they're within a buffer distance of any detected
    machinery/vehicle box this frame. The buffer scales with whichever is bigger — a fraction
    of the frame size, or a fraction of the hazard's OWN size — so a large, close machine gets
    a proportionally bigger real danger radius than a tiny, distant one. Vehicles additionally
    get a larger buffer than stationary machinery, since they can move. Zones are fully
    dynamic — nothing to configure, they just follow wherever equipment is currently detected.
    Returns the hazard's label if the worker is inside a zone, else None."""
    if not hazards:
        return None
    pcx = (person_box[0] + person_box[2]) / 2
    pcy = (person_box[1] + person_box[3]) / 2
    frame_avg = (frame_w + frame_h) / 2
    for hz in hazards:
        x1, y1, x2, y2 = hz["xyxy"]
        hz_diagonal = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        multiplier = HAZARD_BUFFER_MULTIPLIER.get(hz["label"], 1.0)
        buffer_px = max(buffer_ratio * frame_avg, buffer_ratio * hz_diagonal * 0.9) * multiplier
        zx1, zy1, zx2, zy2 = x1 - buffer_px, y1 - buffer_px, x2 + buffer_px, y2 + buffer_px
        if zx1 <= pcx <= zx2 and zy1 <= pcy <= zy2:
            return hz["label"]
    return None

def describe_position(box, frame_w, frame_h):
    """Plain-language description of where in the camera's frame a box is, using a 3x3 grid
    (e.g. 'top-left of frame', 'left side of frame', 'center of frame'). This is the honest
    limit of what a single, uncalibrated camera can tell a supervisor about location — it
    points them to where to look once they're on the right feed, not a real-world coordinate."""
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2

    if cx < frame_w / 3:
        h = "left"
    elif cx < frame_w * 2 / 3:
        h = "center"
    else:
        h = "right"

    if cy < frame_h / 3:
        v = "top"
    elif cy < frame_h * 2 / 3:
        v = "middle"
    else:
        v = "bottom"

    if h == "center" and v == "middle":
        return "center of frame"
    elif v == "middle":
        return f"{h} side of frame"
    elif h == "center":
        return f"{v} of frame"
    else:
        return f"{v}-{h} of frame"

def _get_secret(key):
    """Read a credential from Streamlit secrets first, falling back to an environment variable.
    Never hardcode credentials in source — this is the only place they're read from."""
    try:
        if key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.environ.get(key)

def send_emergency_sms(message):
    """Send an SMS to the supervisor via Twilio. Requires TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN,
    TWILIO_FROM_NUMBER, and SUPERVISOR_PHONE_NUMBER to be set in .streamlit/secrets.toml (or as
    environment variables). Fails silently into the sidebar status log rather than crashing the
    detection loop if anything is missing or the request fails."""
    sid = _get_secret("TWILIO_ACCOUNT_SID")
    token = _get_secret("TWILIO_AUTH_TOKEN")
    from_number = _get_secret("TWILIO_FROM_NUMBER")
    to_number = _get_secret("SUPERVISOR_PHONE_NUMBER")
    stamp = datetime.now().strftime("%H:%M:%S")

    if not all([sid, token, from_number, to_number]):
        st.session_state.sms_status.insert(0, f"[{stamp}] SMS not sent — Twilio not configured.")
        st.session_state.sms_status = st.session_state.sms_status[:10]
        return

    try:
        from twilio.rest import Client
        client = Client(sid, token)
        client.messages.create(body=message, from_=from_number, to=to_number)
        st.session_state.sms_status.insert(0, f"[{stamp}] SMS sent to supervisor.")
    except Exception as e:
        st.session_state.sms_status.insert(0, f"[{stamp}] SMS failed: {e}")
    st.session_state.sms_status = st.session_state.sms_status[:10]

def _generate_siren_wav_base64():
    """Synthesize a loud, continuously-sweeping siren wail (closer to a real emergency horn
    than a flat two-tone beep) as a WAV, base64-encoded, so the emergency banner can play an
    alarm sound with no external audio file dependency."""
    framerate = 8000
    duration = 2.6            # longer blast (was 1.4s)
    n_samples = int(framerate * duration)
    frames = bytearray()

    low_freq, high_freq = 650, 1700   # wider pitch range = more piercing
    sweep_hz = 2.4                     # how fast the pitch wails up/down per second
    amplitude = 0.95                   # near-max volume (was 0.5)
    fade_samples = int(framerate * 0.02)  # 20ms fade to avoid clicks at start/end

    for i in range(n_samples):
        t = i / framerate
        # continuous rising/falling pitch sweep, like a real siren horn, instead of the old
        # hard-switching two-tone beep
        freq = low_freq + (high_freq - low_freq) * (0.5 + 0.5 * math.sin(2 * math.pi * sweep_hz * t))
        raw = math.sin(2 * math.pi * freq * t)
        # mild hard-clipping adds harsh harmonic "edge" closer to a real horn instead of a
        # soft, pure sine tone
        clipped = max(-1.0, min(1.0, raw * 1.7))
        env = 1.0
        if i < fade_samples:
            env = i / fade_samples
        elif i > n_samples - fade_samples:
            env = (n_samples - i) / fade_samples
        val = int(32767 * amplitude * env * clipped)
        frames += struct.pack("<h", val)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        wf.writeframes(bytes(frames))
    return base64.b64encode(buf.getvalue()).decode("ascii")

SIREN_WAV_B64 = _generate_siren_wav_base64()

def log_emergency(worker_id, identity_key, zone_name, source, position=None):
    """Fire an emergency: a worker with an active PPE violation has entered a restricted zone.
    Distinct from a normal violation — pushes a separate alarm banner, plays a siren, texts the
    supervisor, and is also permanently recorded in the violation log as CRITICAL risk. Includes
    a plain-language position within the frame so the supervisor knows where to look."""
    key = f"emergency_{identity_key}_{zone_name}"
    t = time.time()
    if key in st.session_state.emergency_cooldown and t - st.session_state.emergency_cooldown[key] < EMERGENCY_COOLDOWN_SECONDS:
        return
    st.session_state.emergency_cooldown[key] = t

    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    position_txt = position or "position unknown"

    st.session_state.active_emergencies.insert(0, {
        "Timestamp": now, "Worker ID": worker_id, "Zone": zone_name,
        "Position": position_txt, "Source": source,
    })
    st.session_state.active_emergencies = st.session_state.active_emergencies[:10]

    st.session_state.violation_logs.append({
        "Timestamp": now,
        "Worker ID": worker_id,
        "Violation Type": f"EMERGENCY: entered '{zone_name}' without PPE ({position_txt})",
        "Risk": "CRITICAL",
        "Source": source,
    })

    st.toast(f"EMERGENCY: {worker_id} entered '{zone_name}' without PPE — {position_txt}!", icon="⚠")
    send_emergency_sms(
        f"EMERGENCY - Site Safety Control: Worker {worker_id} entered restricted zone "
        f"'{zone_name}' without required PPE. Location in camera view: {position_txt}. "
        f"Source: {source}. Time: {now}."
    )

def _center(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2

def get_or_assign_worker(identity_key):
    """Map a stable identity (tracker ID, or a per-frame fallback) to a persistent Worker ID.
    Each identity is only ever assigned a Worker ID once, so the same person keeps the same
    ID instead of getting a new one on every detection."""
    if "identity_to_worker" not in st.session_state:
        st.session_state.identity_to_worker = {}
    if "next_worker_num" not in st.session_state:
        st.session_state.next_worker_num = 1
    mapping = st.session_state.identity_to_worker
    if identity_key not in mapping:
        mapping[identity_key] = f"W{st.session_state.next_worker_num:03d}"
        st.session_state.next_worker_num += 1
    return mapping[identity_key]

def reset_session_identities():
    """Call this when a brand-new video/webcam session starts, so leftover tracker IDs from a
    previous run don't get mistaken for the same person in a new one. Worker numbering itself
    keeps counting up (via next_worker_num) so IDs are never reused across sessions either."""
    st.session_state.identity_to_worker = {}
    st.session_state.active_tracks = {}
    st.session_state.lost_tracks = {}

IDENTITY_GRACE_SECONDS = 4     # how long we'll try to re-link a briefly-lost person

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def _contains_center(outer, inner):
    icx, icy = _center(inner)
    x1, y1, x2, y2 = outer
    return x1 <= icx <= x2 and y1 <= icy <= y2

def _boxes_close(a, b):
    """True if two boxes are close enough in size/position to plausibly be the same physical
    object even without overlapping (used for re-linking across a brief gap)."""
    if _iou(a, b) > 0:
        return True
    acx, acy = _center(a)
    bcx, bcy = _center(b)
    dist = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
    avg_size = ((a[2] - a[0]) + (a[3] - a[1]) + (b[2] - b[0]) + (b[3] - b[1])) / 4
    return dist < avg_size * 1.2

def _should_merge(a, b):
    """True if two boxes detected in the SAME frame almost certainly belong to the same
    physical person: they overlap at all, one's center sits inside the other (a violation box
    inside a Person box), or they're close enough to be a redundant duplicate detection."""
    if _iou(a, b) > 0:
        return True
    if _contains_center(a, b) or _contains_center(b, a):
        return True
    return _boxes_close(a, b)

def _union_box(boxes):
    return [
        min(b[0] for b in boxes), min(b[1] for b in boxes),
        max(b[2] for b in boxes), max(b[3] for b in boxes),
    ]

def extract_detections(results):
    """Split one frame's YOLO result into Person boxes, PPE-violation boxes, and hazard
    (machinery/vehicle) boxes. Each box is tagged with its tracker ID when available (from
    model.track), so the same physical person or violation instance can be recognized across
    frames."""
    names = model.names
    persons, violations, hazards = [], [], []
    boxes = results[0].boxes
    for box in boxes:
        label = names[int(box.cls[0])]
        xyxy = box.xyxy[0].tolist()
        track_id = int(box.id[0]) if box.id is not None else None
        if label == "Person":
            persons.append({"track_id": track_id, "xyxy": xyxy})
        elif label in VIOLATION_LABELS:
            violations.append({"label": label, "xyxy": xyxy, "track_id": track_id})
        elif label in HAZARD_LABELS:
            hazards.append({"label": label, "xyxy": xyxy, "track_id": track_id})
    return persons, violations, hazards

def cluster_frame_boxes(persons, violations):
    """Group violations under the single physical person they belong to.
    Distinct persons are never merged together (unless IoU > 0.65 duplicate detections).
    Each violation box is assigned to the worker whose body or head contains/is closest to it."""
    # 1. Deduplicate redundant Person boxes for the same physical person
    sorted_persons = sorted(
        persons,
        key=lambda p: (p["xyxy"][2] - p["xyxy"][0]) * (p["xyxy"][3] - p["xyxy"][1]),
        reverse=True,
    )
    distinct_persons = []
    for p in sorted_persons:
        if not any(_iou(p["xyxy"], dp["xyxy"]) > 0.65 for dp in distinct_persons):
            distinct_persons.append(p)

    clusters = [[{"kind": "person", **p}] for p in distinct_persons]

    # 2. Assign each violation to the best matching worker
    unassigned_violations = []
    for v in violations:
        vcx, vcy = _center(v["xyxy"])
        best_cluster_idx = None
        min_dist = float("inf")

        # Priority 1: Check containment inside a person's bounding box
        for idx, cl in enumerate(clusters):
            p_box = cl[0]["xyxy"]
            if p_box[0] <= vcx <= p_box[2] and p_box[1] <= vcy <= p_box[3]:
                pcx, pcy = _center(p_box)
                dist = ((vcx - pcx) ** 2 + (vcy - pcy) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_cluster_idx = idx

        # Priority 2: If slightly outside (e.g. helmet on top of head box), check proximity
        if best_cluster_idx is None:
            for idx, cl in enumerate(clusters):
                p_box = cl[0]["xyxy"]
                pcx, pcy = _center(p_box)
                ph = p_box[3] - p_box[1]
                dist = ((vcx - pcx) ** 2 + (vcy - pcy) ** 2) ** 0.5
                if dist < ph * 0.85 and dist < min_dist:
                    min_dist = dist
                    best_cluster_idx = idx

        if best_cluster_idx is not None:
            clusters[best_cluster_idx].append({"kind": "violation", **v})
        else:
            unassigned_violations.append(v)

    # 3. Handle orphan violations (e.g. head detected without person body)
    if unassigned_violations:
        orphan_groups = []
        for ov in unassigned_violations:
            placed = False
            for og in orphan_groups:
                if _boxes_close(ov["xyxy"], og[0]["xyxy"]):
                    og.append({"kind": "violation", **ov})
                    placed = True
                    break
            if not placed:
                orphan_groups.append([{"kind": "violation", **ov}])
        clusters.extend(orphan_groups)

    return clusters

def resolve_cluster_identity(cluster, frame_tag, cluster_idx):
    """Work out a stable, distinct Worker ID for each worker.
    In image mode, every detected worker gets a unique Worker ID.
    In video/webcam mode, track IDs from the tracker are preserved."""
    active = st.session_state.active_tracks
    lost = st.session_state.lost_tracks
    now = time.time()

    person_items = [it for it in cluster if it["kind"] == "person"]
    track_ids = [p["track_id"] for p in person_items if p["track_id"] is not None]

    identity_key = None
    # Video tracking with persistent track IDs
    for tid in track_ids:
        if tid in active:
            identity_key = active[tid]["identity_key"]
            break

    # Single-image uploads: every worker cluster is uniquely indexed
    if identity_key is None and frame_tag.startswith("img_"):
        identity_key = f"{frame_tag}_w{cluster_idx}"

    # Video stream: re-link briefly lost track only with high spatial overlap
    if identity_key is None:
        cluster_box = _union_box([it["xyxy"] for it in cluster])
        best_lost_id = None
        max_iou = 0.40
        for lost_id, info in list(lost.items()):
            if now - info["lost_at"] <= IDENTITY_GRACE_SECONDS:
                overlap = _iou(info["last_bbox"], cluster_box)
                if overlap > max_iou:
                    max_iou = overlap
                    best_lost_id = lost_id
        if best_lost_id is not None:
            identity_key = lost[best_lost_id]["identity_key"]
            del lost[best_lost_id]

    # Fallback to unique track or cluster key
    if identity_key is None:
        identity_key = f"track_{track_ids[0]}" if track_ids else f"{frame_tag}_c{cluster_idx}"

    for p in person_items:
        if p["track_id"] is not None:
            active[p["track_id"]] = {"last_seen": now, "last_bbox": p["xyxy"], "identity_key": identity_key}

    return identity_key, [tid for tid in track_ids]

def sweep_lost_tracks(seen_track_ids):
    """Move any previously-active track that vanished this frame into the lost pool, and expire
    lost tracks that have been gone too long to safely re-link."""
    active = st.session_state.active_tracks
    lost = st.session_state.lost_tracks
    now = time.time()
    for tid in list(active.keys()):
        if tid not in seen_track_ids:
            lost[tid] = {
                "lost_at": now,
                "last_bbox": active[tid]["last_bbox"],
                "identity_key": active[tid]["identity_key"],
            }
            del active[tid]
    for lid in list(lost.keys()):
        if now - lost[lid]["lost_at"] > IDENTITY_GRACE_SECONDS:
            del lost[lid]

ZONE_DEBOUNCE_FRAMES = 2  # for live video/webcam: require this many consecutive frames of
                          # actual zone entry before firing, so one jittery frame can't trigger
                          # an emergency alone. Image mode (single snapshot) always confirms
                          # immediately since there's only one frame to check in the first place.

def process_violations(results, source, frame_tag, frame_w=None, frame_h=None):
    """Log every PPE violation in this frame under the stable Worker ID of the physical worker
    it belongs to, using frame-wide clustering so redundant/noisy Person detections can never
    split one person's violations across multiple Worker IDs. If a worker with an active
    violation is near detected machinery/vehicle for enough consecutive frames, also fires an
    emergency alert.

    Returns a list of per-worker status entries — one for EVERY tracked person, whether or not
    they have a violation — so the caller can draw a live Worker ID tag on screen for everyone,
    not just people currently in violation."""
    if "active_tracks" not in st.session_state:
        st.session_state.active_tracks = {}
    if "lost_tracks" not in st.session_state:
        st.session_state.lost_tracks = {}
    if "zone_streak" not in st.session_state:
        st.session_state.zone_streak = {}

    persons, violations, hazards = extract_detections(results)
    clusters = cluster_frame_boxes(persons, violations)

    seen_track_ids = set()
    worker_annotations = []
    is_stream = not frame_tag.startswith("img_")  # False for single-image uploads

    for idx, cluster in enumerate(clusters):
        identity_key, track_ids = resolve_cluster_identity(cluster, frame_tag, idx)
        seen_track_ids.update(track_ids)
        worker_id = get_or_assign_worker(identity_key)
        cluster_violations = [v for v in cluster if v["kind"] == "violation"]
        unique_violation_types = sorted(set(v["label"] for v in cluster_violations))
        for v_type in unique_violation_types:
            log_violation(v_type, source, worker_id, identity_key)

        person_items = [it for it in cluster if it["kind"] == "person"]
        if person_items:
            person_box = _union_box([p["xyxy"] for p in person_items])
        elif cluster_violations:
            # No Person box was detected for this cluster (common for partial/rear/side views
            # where the body detector misses but the head-region violation detector still
            # fires). A head/violation box alone sits near the TOP of the person, which badly
            # underestimates where their actual body is relative to ground-level machinery —
            # so extrapolate a rough full-body region downward from it instead of using the
            # head box's position directly.
            hx1, hy1, hx2, hy2 = _union_box([v["xyxy"] for v in cluster_violations])
            head_h = hy2 - hy1
            person_box = [hx1, hy1, hx2, hy2 + head_h * 6]  # head ≈ 1/7 of standing height
        else:
            person_box = None

        if person_box is not None:
            missing = sorted(set(v["label"].replace("NO-", "") for v in cluster_violations))
            worker_annotations.append({
                "worker_id": worker_id,
                "box": person_box,
                "missing": missing,  # empty list = fully compliant, everything present
            })

            in_zone_hazard = None
            if cluster_violations and frame_w and frame_h and hazards:
                buffer_ratio = st.session_state.get("danger_buffer_ratio", DEFAULT_DANGER_BUFFER_RATIO)
                in_zone_hazard = in_danger_zone(person_box, hazards, frame_w, frame_h, buffer_ratio)

            if in_zone_hazard:
                # single images confirm immediately (only one frame exists to check); live
                # streams need the same worker+hazard combo seen on consecutive frames first
                st.session_state.zone_streak[identity_key] = (
                    ZONE_DEBOUNCE_FRAMES if not is_stream
                    else st.session_state.zone_streak.get(identity_key, 0) + 1
                )
                if st.session_state.zone_streak[identity_key] >= ZONE_DEBOUNCE_FRAMES:
                    position = describe_position(person_box, frame_w, frame_h)
                    log_emergency(worker_id, identity_key, f"Near {in_zone_hazard}", source, position)
            else:
                st.session_state.zone_streak[identity_key] = 0

    sweep_lost_tracks(seen_track_ids)
    return worker_annotations

def draw_worker_overlay(annotated_bgr, worker_annotations):
    """Draw a persistent Worker ID tag under every tracked person, live — green 'COMPLIANT'
    for workers with everything on, red listing exactly what's missing otherwise. Runs for
    every detected person, not just people currently in violation."""
    for w in worker_annotations:
        x1, y1, x2, y2 = [int(v) for v in w["box"]]
        if w["missing"]:
            label = f'{w["worker_id"]} - MISSING: {", ".join(w["missing"])}'
            color = (0, 0, 230)  # BGR red
        else:
            label = f'{w["worker_id"]} - COMPLIANT'
            color = (60, 180, 75)  # BGR green

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        frame_h, frame_w = annotated_bgr.shape[:2]
        tag_y1 = min(y2 + 4, frame_h - th - 10)
        tag_y1 = max(tag_y1, 0)
        tag_y2 = tag_y1 + th + 10
        tag_x1 = max(x1, 0)
        tag_x2 = min(tag_x1 + tw + 10, frame_w)

        cv2.rectangle(annotated_bgr, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
        cv2.putText(annotated_bgr, label, (tag_x1 + 5, tag_y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return annotated_bgr



def log_violation(label, source, worker_id, identity_key):
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if "last_log" not in st.session_state:
        st.session_state.last_log = {}
    key = f"{worker_id}_{label}"
    t = time.time()
    is_stream = not str(source).lower().startswith("image")
    if is_stream and key in st.session_state.last_log and t - st.session_state.last_log[key] < VIOLATION_COOLDOWN_SECONDS:
        return
    st.session_state.last_log[key] = t
    st.session_state.violation_logs.append({
        "Timestamp":now,
        "Worker ID":worker_id,
        "Violation Type":label,
        "Risk":"HIGH",
        "Source":source
    })

    # Push a live alert for the supervisor dashboard
    st.session_state.active_alerts.insert(0, {
        "Timestamp": now,
        "Worker ID": worker_id,
        "Violation Type": label,
        "Source": source
    })
    # keep only the most recent 20 alerts on screen
    st.session_state.active_alerts = st.session_state.active_alerts[:20]

    # Pop-up toast notification (works no matter which tab is open)
    st.toast(f"⚠️ {label.replace('NO-', 'Missing ')} detected ({source})", icon="🚨")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root {
    --steel-900:#15171B;
    --steel-800:#1D2026;
    --steel-700:#262A31;
    --steel-600:#343941;
    --steel-line: rgba(255,255,255,0.08);
    --concrete-100:#EDEAE2;
    --concrete-200:#C9C4B8;
    --safety-orange:#FF6A13;
    --safety-orange-dark:#D9560A;
    --hazard-yellow:#FFC300;
    --signal-red:#E63946;
    --signal-red-dark:#A82535;
    --blueprint:#4A7FB5;
}

html, body, [class*="css"], .stMarkdown, p, span, label {
    font-family: 'Inter', sans-serif;
}

/* ---------- App shell: dark steel + faint blueprint grid ---------- */
.stApp {
    background:
        repeating-linear-gradient(0deg, rgba(74,127,181,0.05) 0px, rgba(74,127,181,0.05) 1px, transparent 1px, transparent 42px),
        repeating-linear-gradient(90deg, rgba(74,127,181,0.05) 0px, rgba(74,127,181,0.05) 1px, transparent 1px, transparent 42px),
        linear-gradient(180deg, var(--steel-900), #0F1114);
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"] { background: transparent; }
[data-testid="stHeader"] { background: transparent; }

/* ---------- Sidebar: riveted steel control panel ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--steel-800) 0%, var(--steel-900) 100%);
    border-right: 1px solid var(--steel-line);
}
[data-testid="stSidebar"] * { color: var(--concrete-100) !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h1 {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 1.05rem !important;
    color: var(--hazard-yellow) !important;
    border-bottom: 3px solid var(--hazard-yellow);
    padding-bottom: 10px;
}

/* ---------- Hazard stripe divider (signature element, used only near alerts) ---------- */
.hazard-stripe {
    height: 7px;
    border-radius: 3px;
    background: repeating-linear-gradient(135deg, var(--hazard-yellow) 0 16px, #14161A 16px 32px);
    margin: 0 0 16px 0;
}

/* ---------- Entry screen: crossed hazard tape + CAUTION tag ---------- */
.hazard-frame-wrap {
    position: relative;
    max-width: 640px;
    margin: 70px auto 0 auto;
    padding: 60px 10px;
}
.hazard-frame-wrap .tape {
    position: absolute;
    top: 50%;
    left: 50%;
    width: 150%;
    height: 30px;
    background: repeating-linear-gradient(135deg, var(--hazard-yellow) 0 16px, #14161A 16px 32px);
    box-shadow: 0 10px 26px rgba(0,0,0,0.55);
    z-index: 1;
}
.hazard-frame-wrap .tape-a { transform: translate(-50%, -50%) rotate(19deg); }
.hazard-frame-wrap .tape-b { transform: translate(-50%, -50%) rotate(-19deg); }
.site-card {
    position: relative;
    z-index: 2;
    background: linear-gradient(180deg, var(--steel-800), var(--steel-900));
    border: 1px solid var(--steel-line);
    border-radius: 8px;
    padding: 56px 40px 40px 40px;
    color: var(--concrete-100);
    text-align: center;
    box-shadow: 0 20px 60px rgba(0,0,0,0.55);
}
.caution-tag {
    position: absolute;
    top: -22px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 4;
    border-radius: 4px;
    overflow: hidden;
    box-shadow: 0 10px 26px rgba(0,0,0,0.55);
    white-space: nowrap;
}
.caution-stripe {
    height: 6px;
    background: repeating-linear-gradient(135deg, var(--hazard-yellow) 0 10px, #14161A 10px 20px);
}
.caution-body {
    background: var(--safety-orange);
    color: #14161A;
    font-family: 'Oswald', sans-serif;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-size: 1.05rem;
    padding: 9px 30px;
}
.site-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-size: 0.75rem;
    color: var(--safety-orange);
    margin: 8px 0 14px 0;
}
.site-card h1 {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    font-weight: 700;
    font-size: 2.6rem;
    margin: 0;
    color: var(--concrete-100);
}
.site-card .site-tagline {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: var(--concrete-200);
    margin: 10px 0 0 0;
    letter-spacing: 0.03em;
}
.site-card .site-desc {
    max-width: 440px;
    margin: 22px auto 0 auto;
    font-size: 0.98rem;
    line-height: 1.6;
    color: var(--concrete-100);
    opacity: 0.85;
}

/* ---------- Buttons: stenciled push-button ---------- */
div.stButton > button {
    background: var(--safety-orange);
    color: #1A1200;
    border: none;
    border-radius: 6px;
    padding: 0.8rem 1.4rem;
    font-family: 'Oswald', sans-serif;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    width: 100%;
    box-shadow: inset 0 -3px 0 rgba(0,0,0,0.25), 0 8px 18px rgba(0,0,0,0.4);
}
div.stButton > button:hover {
    background: var(--safety-orange-dark);
    color: white;
}
div.stButton > button:active {
    box-shadow: inset 0 2px 0 rgba(0,0,0,0.3);
}

/* ---------- Module nav: illuminated toggle switches ---------- */
[data-testid="stSidebar"] div.stButton > button[kind="secondary"] {
    background: var(--steel-700);
    color: var(--concrete-200);
    text-align: left;
    justify-content: flex-start;
    border: 1px solid var(--steel-line);
    border-left: 4px solid var(--steel-600);
    border-radius: 6px;
    box-shadow: none;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: none;
    letter-spacing: 0.02em;
    font-weight: 500;
    padding: 0.65rem 1rem;
    margin-bottom: 6px;
}
[data-testid="stSidebar"] div.stButton > button[kind="secondary"]:hover {
    background: var(--steel-600);
    border-left: 4px solid var(--safety-orange);
    color: var(--concrete-100);
}
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, var(--steel-700), var(--steel-800));
    color: var(--hazard-yellow);
    text-align: left;
    justify-content: flex-start;
    border: 1px solid rgba(255,195,0,0.35);
    border-left: 4px solid var(--hazard-yellow);
    border-radius: 6px;
    box-shadow: 0 0 0 1px rgba(255,195,0,0.15), 0 0 18px rgba(255,195,0,0.25);
    font-family: 'IBM Plex Mono', monospace;
    text-transform: none;
    letter-spacing: 0.02em;
    font-weight: 600;
    padding: 0.65rem 1rem;
    margin-bottom: 6px;
}
[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--steel-600), var(--steel-700));
    color: var(--hazard-yellow);
}

/* ---------- Main-screen module nav tiles ---------- */
.nav-label {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.75rem;
    color: var(--concrete-200);
    margin: 4px 0 10px 0;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--steel-700);
    border: 1px solid var(--steel-line) !important;
    border-radius: 8px !important;
    padding: 2px !important;
    transition: border-color 0.15s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(button[kind="primary"]) {
    border: 1px solid rgba(255,195,0,0.5) !important;
    box-shadow: 0 0 18px rgba(255,195,0,0.2);
}
[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"] {
    background: transparent !important;
    color: var(--concrete-200) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Oswald', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600 !important;
    padding: 1.1rem 0.6rem !important;
    box-shadow: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"] button[kind="secondary"]:hover {
    background: var(--steel-600) !important;
    color: var(--concrete-100) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] button[kind="primary"] {
    background: linear-gradient(135deg, var(--steel-700), var(--steel-800)) !important;
    color: var(--hazard-yellow) !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Oswald', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700 !important;
    padding: 1.1rem 0.6rem !important;
    box-shadow: none !important;
}
[data-testid="stVerticalBlockBorderWrapper"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--steel-600), var(--steel-700)) !important;
    color: var(--hazard-yellow) !important;
}

/* ---------- File uploader: dark steel dropzone, hazard-orange dashed border, centered stack ---------- */
div[data-testid="stFileUploader"] {
    background: transparent;
    padding: 0;
    border: none;
    box-shadow: none;
}
div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] label p {
    font-family: 'Oswald', sans-serif !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    color: var(--hazard-yellow) !important;
    margin-bottom: 12px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: var(--steel-700) !important;
    border: 2px dashed var(--safety-orange) !important;
    border-radius: 12px !important;
    padding: 44px 20px !important;
    min-height: 200px;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    gap: 8px !important;
    box-shadow: 0 10px 26px rgba(0,0,0,0.35);
    transition: border-color 0.15s ease, background 0.15s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--hazard-yellow) !important;
    background: var(--steel-600) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    background: transparent !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 10px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] * {
    background: transparent !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    width: 46px !important;
    height: 46px !important;
    fill: var(--safety-orange) !important;
    color: var(--safety-orange) !important;
    margin-bottom: 4px;
}
[data-testid="stFileUploaderDropzone"] span {
    color: var(--concrete-100) !important;
    font-family: 'Oswald', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.02rem !important;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}
[data-testid="stFileUploaderDropzone"] small {
    color: var(--concrete-200) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    letter-spacing: 0.02em;
    font-size: 0.78rem !important;
}
[data-testid="stFileUploaderDropzone"] button {
    margin-top: 10px !important;
    background: var(--safety-orange) !important;
    color: #1A1200 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Oswald', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.3rem !important;
    box-shadow: inset 0 -3px 0 rgba(0,0,0,0.25) !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: var(--safety-orange-dark) !important;
    color: white !important;
}
[data-testid="stFileUploaderFile"] {
    background: var(--steel-800) !important;
    border: 1px solid var(--steel-line) !important;
    border-radius: 6px !important;
    color: var(--concrete-100) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    margin-top: 10px;
}
[data-testid="stFileUploaderFile"] small {
    color: var(--concrete-200) !important;
}

div[data-testid="stRadio"] {
    background: var(--steel-700);
    padding: 16px;
    border-radius: 10px;
    border: 1px solid var(--steel-line);
    box-shadow: 0 10px 26px rgba(0,0,0,0.3);
}
div[data-testid="stRadio"] label {
    color: var(--concrete-100) !important;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

img {
    border-radius: 8px;
    box-shadow: 0 12px 30px rgba(0,0,0,0.4);
    border: 1px solid var(--steel-line);
}

/* ---------- Info / status boxes styled like a posted site notice ---------- */
[data-testid="stAlert"] {
    background: var(--steel-700) !important;
    border: 1px dashed var(--concrete-200) !important;
    border-radius: 8px !important;
    color: var(--concrete-100) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
}

/* ---------- Metrics: stamped data plates ---------- */
[data-testid="stMetric"] {
    background: var(--steel-700);
    border: 1px solid var(--steel-line);
    border-left: 4px solid var(--safety-orange);
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--concrete-200) !important;
}
[data-testid="stMetricValue"] {
    color: var(--hazard-yellow) !important;
    font-family: 'Oswald', sans-serif;
}

/* ---------- Dataframe styling ---------- */
[data-testid="stDataFrame"] {
    border: 1px solid var(--steel-line);
    border-radius: 8px;
    overflow: hidden;
}

/* ---------- Slider: hazard orange/yellow instead of Streamlit's default red ---------- */
[data-testid="stSlider"] label p {
    color: var(--concrete-100) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: var(--hazard-yellow) !important;
    border-color: var(--hazard-yellow) !important;
    box-shadow: 0 0 0 4px rgba(255, 195, 0, 0.2) !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: var(--concrete-200) !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
[data-testid="stSliderThumbValue"],
[data-testid="stSlider"] [data-testid="stThumbValue"] {
    color: var(--hazard-yellow) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 700 !important;
}
/* Fallback: hue-shift Streamlit's red accent (~0deg) toward safety-orange (~22deg) so the
   filled track/focus-ring recolor correctly even if the internal slider markup changes */
[data-testid="stSlider"] [data-baseweb="slider"] {
    filter: hue-rotate(22deg) saturate(1.15);
}

/* ---------- Site header panel ---------- */
.site-header {
    background: linear-gradient(135deg, var(--steel-800), var(--steel-700));
    border: 1px solid var(--steel-line);
    border-radius: 10px;
    padding: 26px 32px 22px 32px;
    margin-bottom: 4px;
    box-shadow: 0 14px 34px rgba(0,0,0,0.45);
    position: relative;
}
.site-header::before, .site-header::after {
    content: "";
    position: absolute;
    top: 14px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #6b7280, #2b2f36);
    box-shadow: 0 1px 2px rgba(0,0,0,0.6);
}
.site-header::before { left: 14px; }
.site-header::after { right: 14px; }
.site-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    font-size: 0.7rem;
    color: var(--hazard-yellow);
    margin: 0 0 6px 0;
}
.site-header h1 {
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.01em;
    font-weight: 700;
    font-size: 2rem;
    color: var(--concrete-100);
    margin: 0 0 6px 0;
}
.site-header p {
    color: var(--concrete-200);
    font-size: 0.96rem;
    margin: 0;
}

/* ---------- Alert banner: hazard-tape framed panel (signature) ---------- */
.alert-banner {
    background: linear-gradient(135deg, var(--signal-red-dark), var(--signal-red));
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 10px;
    padding: 14px 18px 16px 18px;
    margin-bottom: 4px;
    box-shadow: 0 12px 28px rgba(230, 57, 70, 0.35);
    animation: pulseAlert 1.8s ease-in-out infinite;
}
.alert-banner-header {
    color: white;
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 8px;
}
.alert-item {
    color: #FFE3E5;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.84rem;
    padding: 4px 0;
    border-bottom: 1px dashed rgba(255,255,255,0.18);
}
.alert-item:last-child { border-bottom: none; }
.alert-badge {
    display: inline-block;
    background: #14161A;
    color: var(--hazard-yellow);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 700;
    margin-left: 8px;
    font-family: 'IBM Plex Mono', monospace;
}
@keyframes pulseAlert {
    0% { box-shadow: 0 12px 28px rgba(230, 57, 70, 0.3); }
    50% { box-shadow: 0 12px 36px rgba(230, 57, 70, 0.65); }
    100% { box-shadow: 0 12px 28px rgba(230, 57, 70, 0.3); }
}

/* ---------- Emergency banner: worker w/o PPE in a restricted zone (max urgency) ---------- */
.emergency-banner {
    background: repeating-linear-gradient(
        135deg,
        var(--signal-red-dark) 0 22px,
        #7A0E17 22px 44px
    );
    border: 2px solid var(--hazard-yellow);
    border-radius: 10px;
    padding: 16px 20px 18px 20px;
    margin-bottom: 4px;
    box-shadow: 0 0 0 4px rgba(230,57,70,0.25), 0 14px 34px rgba(230, 57, 70, 0.55);
    animation: pulseEmergency 0.9s ease-in-out infinite;
}
.emergency-banner-header {
    color: white;
    font-family: 'Oswald', sans-serif;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 700;
    font-size: 1.15rem;
    margin-bottom: 10px;
    text-shadow: 0 2px 6px rgba(0,0,0,0.5);
}
.emergency-item {
    color: #FFF3D6;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.88rem;
    font-weight: 600;
    padding: 5px 0;
    border-bottom: 1px dashed rgba(255,255,255,0.25);
}
.emergency-item:last-child { border-bottom: none; }
.emergency-badge {
    display: inline-block;
    background: #14161A;
    color: var(--hazard-yellow);
    border: 1px solid var(--hazard-yellow);
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 700;
    margin-left: 10px;
    font-family: 'IBM Plex Mono', monospace;
}
@keyframes pulseEmergency {
    0%   { box-shadow: 0 0 0 4px rgba(230,57,70,0.25), 0 14px 34px rgba(230, 57, 70, 0.5); transform: scale(1); }
    50%  { box-shadow: 0 0 0 8px rgba(255,195,0,0.35), 0 14px 44px rgba(230, 57, 70, 0.85); transform: scale(1.01); }
    100% { box-shadow: 0 0 0 4px rgba(230,57,70,0.25), 0 14px 34px rgba(230, 57, 70, 0.5); transform: scale(1); }
}

/* ---------- Full-site red alarm glow: covers the whole viewport while an emergency is
   active. pointer-events:none so it never blocks clicks/scrolling underneath it. ---------- */
@keyframes emergencyScreenFlash {
    0%   { opacity: 0.25; }
    50%  { opacity: 0.75; }
    100% { opacity: 0.25; }
}
.emergency-flash-overlay {
    position: fixed;
    inset: 0;
    z-index: 999999;
    pointer-events: none;
    background: radial-gradient(ellipse at center, rgba(230,57,70,0) 30%, rgba(230,57,70,0.55) 75%, rgba(122,14,23,0.85) 100%);
    animation: emergencyScreenFlash 0.6s ease-in-out infinite;
}

/* ---------- Restriction zone preview ---------- */
.zone-preview {
    position: relative;
    width: 100%;
    aspect-ratio: 16 / 9;
    background: var(--steel-800);
    border: 1px solid var(--steel-line);
    border-radius: 8px;
    overflow: hidden;
    background-image:
        repeating-linear-gradient(0deg, rgba(74,127,181,0.08) 0px, rgba(74,127,181,0.08) 1px, transparent 1px, transparent 10%),
        repeating-linear-gradient(90deg, rgba(74,127,181,0.08) 0px, rgba(74,127,181,0.08) 1px, transparent 1px, transparent 10%);
}
.zone-preview-label {
    position: absolute;
    top: 8px;
    left: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--concrete-200);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    opacity: 0.7;
}
.zone-box {
    position: absolute;
    background: repeating-linear-gradient(135deg, rgba(255,195,0,0.18) 0 10px, rgba(230,57,70,0.18) 10px 20px);
    border: 2px dashed var(--hazard-yellow);
    border-radius: 4px;
}
.zone-box-label {
    position: absolute;
    top: -22px;
    left: 0;
    background: var(--signal-red);
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 4px;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)


def render_emergency(max_items=5):
    """Render the emergency banner whenever a worker without PPE has entered a restricted
    zone. Plays the siren audio only when the top event is new, not on every re-render (this
    gets called ~30x/second during live video/webcam, so without this guard it would replay
    the siren every frame)."""
    emergencies = st.session_state.get("active_emergencies", [])
    if not emergencies:
        return

    top = emergencies[0]
    signature = f"{top['Timestamp']}_{top['Worker ID']}_{top['Zone']}"
    play_audio = st.session_state.get("last_emergency_signature") != signature
    if play_audio:
        st.session_state.last_emergency_signature = signature

    items_html = ""
    for e in emergencies[:max_items]:
        items_html += (
            f'<div class="emergency-item">'
            f'⏱ {e["Timestamp"]} &nbsp;|&nbsp; WORKER {e["Worker ID"]} &nbsp;|&nbsp; '
            f'ZONE: <b>{e["Zone"]}</b> &nbsp;|&nbsp; LOCATION: <b>{e.get("Position", "unknown")}</b> '
            f'&nbsp;|&nbsp; {e["Source"]}'
            f'</div>'
        )

    audio_html = (
        f'<audio autoplay loop><source src="data:audio/wav;base64,{SIREN_WAV_B64}" type="audio/wav"></audio>'
        if play_audio else ""
    )

    st.markdown(f"""
    <div class="emergency-flash-overlay"></div>
    <div class="emergency-banner">
        <div class="emergency-banner-header">‼ EMERGENCY — WORKER IN RESTRICTED ZONE WITHOUT PPE
            <span class="emergency-badge">{len(emergencies)} ACTIVE</span>
        </div>
        {items_html}
    </div>
    {audio_html}
    """, unsafe_allow_html=True)

def render_alerts(max_items=5):
    """Render a pulsing alert banner on the supervisor dashboard whenever
    there are active PPE violations that haven't been cleared yet."""
    alerts = st.session_state.get("active_alerts", [])
    if not alerts:
        return

    items_html = ""
    for a in alerts[:max_items]:
        clean_label = a["Violation Type"].replace("NO-", "Missing ")
        items_html += (
            f'<div class="alert-item">'
            f'⏱ {a["Timestamp"]} &nbsp;|&nbsp; 👷 {a["Worker ID"]} &nbsp;|&nbsp; '
            f'🚫 <b>{clean_label}</b> &nbsp;|&nbsp; 📍 {a["Source"]}'
            f'</div>'
        )

    st.markdown(f"""
    <div class="hazard-stripe"></div>
    <div class="alert-banner">
        <div class="alert-banner-header">⚠ PPE VIOLATION — SITE HAZARD
            <span class="alert-badge">{len(alerts)} ACTIVE</span>
        </div>
        {items_html}
    </div>
    """, unsafe_allow_html=True)
    # Note: the "Clear Alerts" button lives in the sidebar (render_alerts() may be
    # called repeatedly inside video/webcam loops, and a button can't be redrawn
    # with the same key more than once per script run).

if st.session_state.page == "home":
    st.markdown("""
    <div class="hazard-frame-wrap">
        <div class="tape tape-a"></div>
        <div class="tape tape-b"></div>
        <div class="site-card">
            <div class="caution-tag">
                <div class="caution-stripe"></div>
                <div class="caution-body"><span>⚠</span> CAUTION <span>⚠</span></div>
                <div class="caution-stripe"></div>
            </div>
            <p class="site-eyebrow">Restricted Access &nbsp;//&nbsp; Authorized Personnel Only</p>
            <h1>Site Safety Control</h1>
            <p class="site-tagline">AI-Based PPE Compliance &amp; Violation Monitoring System</p>
            <p class="site-desc">This system scans image, video, and live camera feeds for hard hats,
            masks, and safety vests, and flags non-compliance to site supervisors
            in real time. Proceed to check in and start monitoring.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 0.8, 1])
    with c2:
        if st.button("Enter Site", use_container_width=True):
            st.session_state.page = "upload"
            st.rerun()

else:
    st.markdown("""
    <div class="site-header">
        <p class="site-eyebrow">Live Monitoring &nbsp;//&nbsp; PPE Compliance Console</p>
        <h1>Site Safety Control</h1>
        <p>Upload an image or video, or start the live camera feed to scan for hard hats, masks, and safety vests.</p>
    </div>
    <div style="height: 20px;"></div>
    """, unsafe_allow_html=True)

    emergency_slot = st.empty()
    with emergency_slot.container():
        render_emergency()

    alert_slot = st.empty()
    with alert_slot.container():
        render_alerts()

    with st.sidebar:
        st.header("Control Panel")

        n_emergency = len(st.session_state.active_emergencies)
        if n_emergency:
            st.markdown(
                f'<div style="background:linear-gradient(135deg,#7A0E17,#E63946);color:white;'
                f'padding:10px 14px;border-radius:6px;font-weight:700;text-align:center;'
                f'margin-bottom:10px;font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;'
                f'letter-spacing:0.03em;border:1px solid #FFC300;">'
                f'‼ {n_emergency} EMERGENCY{"S" if n_emergency != 1 else ""}</div>',
                unsafe_allow_html=True
            )
            if st.button("✕  Clear Emergencies", key="clear_emergencies_btn", use_container_width=True):
                st.session_state.active_emergencies = []
                st.rerun()

        n_active = len(st.session_state.active_alerts)
        if n_active:
            st.markdown(
                f'<div style="background:#E63946;color:white;padding:10px 14px;'
                f'border-radius:6px;font-weight:700;text-align:center;margin-bottom:10px;'
                f'font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;letter-spacing:0.03em;'
                f'border-left:4px solid #FFC300;">'
                f'⚠ {n_active} ACTIVE ALERT{"S" if n_active != 1 else ""}</div>',
                unsafe_allow_html=True
            )
            if st.button("✕  Clear Alerts", key="clear_alerts_btn", use_container_width=True):
                st.session_state.active_alerts = []
                st.rerun()
        else:
            st.markdown(
                '<p style="font-family:\'IBM Plex Mono\', monospace; font-size:0.8rem; '
                'color:#C9C4B8;">No active alerts.</p>',
                unsafe_allow_html=True
            )

        if st.session_state.sms_status:
            with st.expander("SMS Status", expanded=False):
                for line in st.session_state.sms_status[:5]:
                    st.markdown(
                        f'<p style="font-family:\'IBM Plex Mono\', monospace; font-size:0.72rem; '
                        f'color:#C9C4B8; margin:2px 0;">{line}</p>',
                        unsafe_allow_html=True
                    )

        st.markdown(
            '<p style="font-family:\'IBM Plex Mono\', monospace; text-transform:uppercase; '
            'letter-spacing:0.1em; font-size:0.72rem; color:#C9C4B8; margin:16px 0 4px 0; '
            'border-top:1px solid rgba(255,255,255,0.1); padding-top:14px;">Detection Sensitivity</p>',
            unsafe_allow_html=True
        )
        conf_threshold = st.slider(
            "Confidence threshold", 0.10, 0.90, 0.40, 0.05,
            help="Higher = fewer, more confident detections (less noise, but may miss some "
                 "violations). Lower = catches more, but more false positives/misclassifications."
        )
        inference_size = st.select_slider(
            "Inference resolution",
            options=[640, 800, 960, 1120, 1280],
            value=960,
            help="Higher = the model sees more detail on small/distant people and PPE — helps "
                 "in crowded scenes. Cost: noticeably slower, especially on live webcam."
        )
        nms_iou = st.slider(
            "NMS overlap tolerance (IoU)", 0.30, 0.90, 0.75, 0.05,
            help="Higher = more tolerant of overlapping boxes, so people standing close "
                 "together in a crowd are less likely to have one of them suppressed as a "
                 "'duplicate'. Lower = cleaner single-box-per-object output, but more likely "
                 "to lose a real detection in a crowd."
        )

    if "nav_option" not in st.session_state:
        st.session_state.nav_option = "Image"

    st.markdown('<p class="nav-label">Select Module</p>', unsafe_allow_html=True)

    modules = ["Image", "Video", "Webcam", "Violation Logs", "Danger Zones"]
    nav_cols = st.columns(len(modules))
    for col, name in zip(nav_cols, modules):
        with col:
            with st.container(border=True):
                is_active = st.session_state.nav_option == name
                if st.button(
                    name,
                    key=f"nav_{name}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state.nav_option = name
                    st.rerun()

    option = st.session_state.nav_option

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    st.info("SELECT A MODULE ABOVE, THEN UPLOAD OR START THE LIVE FEED.")
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


    if option == "Violation Logs":
        st.markdown("""
        <p style="font-family:'Oswald',sans-serif;text-transform:uppercase;
        letter-spacing:0.05em;color:#EDEAE2;font-size:1.4rem;font-weight:700;
        margin-bottom:14px;">Violation Log — Site Record</p>
        """, unsafe_allow_html=True)
        logs=st.session_state.violation_logs
        if logs:
            df=pd.DataFrame(logs)
            c1,c2,c3=st.columns(3)
            c1.metric("Total Violations",len(df))
            risk=st.selectbox("Filter Risk",["All"]+sorted(df["Risk"].unique().tolist()))
            vio=st.selectbox("Filter Violation",["All"]+sorted(df["Violation Type"].unique().tolist()))
            search=st.text_input("Search Worker ID")
            if risk!="All": df=df[df["Risk"]==risk]
            if vio!="All": df=df[df["Violation Type"]==vio]
            if search: df=df[df["Worker ID"].str.contains(search,case=False)]
            st.dataframe(df,use_container_width=True)
            st.download_button("Download Logs CSV",df.to_csv(index=False),"violation_logs.csv","text/csv")
        else:
            st.info("No violations recorded yet.")

    elif option == "Danger Zones":
        st.markdown("""
        <p style="font-family:'Oswald',sans-serif;text-transform:uppercase;
        letter-spacing:0.05em;color:#EDEAE2;font-size:1.4rem;font-weight:700;
        margin-bottom:6px;">Danger Zones</p>
        <p style="font-family:'IBM Plex Mono',monospace;font-size:0.82rem;color:#C9C4B8;
        margin-bottom:18px;">Danger zones are fully automatic — they're drawn around whatever
        machinery or vehicles the model detects in the current frame, no manual setup required.
        A worker with an active PPE violation who gets within the buffer distance below
        triggers an emergency alert.</p>
        """, unsafe_allow_html=True)

        setting_col, preview_col = st.columns([1, 1.3])

        with setting_col:
            st.markdown('<p class="nav-label">Buffer Distance</p>', unsafe_allow_html=True)
            buffer_pct = st.slider(
                "How close is 'too close'? (% of frame size)",
                5, 40, int(st.session_state.danger_buffer_ratio * 100), 1,
                help="A larger buffer triggers emergencies from further away from machinery/vehicles; "
                     "a smaller buffer only triggers when a worker is right next to it."
            )
            new_ratio = buffer_pct / 100
            if new_ratio != st.session_state.danger_buffer_ratio:
                st.session_state.danger_buffer_ratio = new_ratio
                save_danger_buffer_ratio(new_ratio)

            st.markdown(
                '<p style="font-family:\'IBM Plex Mono\', monospace; font-size:0.78rem; '
                'color:#C9C4B8; margin-top:14px;">Zones are detected live from the '
                '<b style="color:#EDEAE2;">machinery</b> and <b style="color:#EDEAE2;">vehicle</b> '
                'classes your model already recognizes — nothing to draw or configure per camera.</p>',
                unsafe_allow_html=True
            )

        with preview_col:
            st.markdown('<p class="nav-label">Illustration</p>', unsafe_allow_html=True)
            # illustrative example only — not a live feed; shows how the buffer scales
            hz_x1, hz_y1, hz_x2, hz_y2 = 35, 32, 65, 68
            buf = buffer_pct
            zone_x1 = max(0, hz_x1 - buf)
            zone_y1 = max(0, hz_y1 - buf)
            zone_x2 = min(100, hz_x2 + buf)
            zone_y2 = min(100, hz_y2 + buf)
            st.markdown(f"""
            <div class="zone-preview">
                <div class="zone-preview-label">Example — not your live feed</div>
                <div class="zone-box" style="left:{zone_x1}%;top:{zone_y1}%;width:{zone_x2-zone_x1}%;height:{zone_y2-zone_y1}%;">
                    <div class="zone-box-label">Danger Zone ({buffer_pct}% buffer)</div>
                </div>
                <div style="position:absolute;left:{hz_x1}%;top:{hz_y1}%;width:{hz_x2-hz_x1}%;height:{hz_y2-hz_y1}%;
                     background:var(--steel-600);border:2px solid var(--concrete-200);border-radius:4px;
                     display:flex;align-items:center;justify-content:center;">
                    <span style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:var(--concrete-100);
                     text-transform:uppercase;letter-spacing:0.05em;">Machinery</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("The gray box is detected machinery/a vehicle; the red hazard-striped area is "
                       "the live danger zone around it, sized by your buffer setting above.")

        with st.expander("SMS setup (Twilio) — one-time, do this on your machine, not here"):
            st.markdown("""
            Create a file at `.streamlit/secrets.toml` in your project folder (**never commit
            or share this file**) with:

            ```
            TWILIO_ACCOUNT_SID = "your_account_sid"
            TWILIO_AUTH_TOKEN = "your_auth_token"
            TWILIO_FROM_NUMBER = "+1XXXXXXXXXX"
            SUPERVISOR_PHONE_NUMBER = "+1XXXXXXXXXX"
            ```

            You'll also need to install the Twilio SDK:
            ```
            pip install twilio
            ```

            Restart the app after adding secrets. If they're missing, emergencies still fire
            in-app (banner + siren) — you'll just see "SMS not sent — Twilio not configured"
            in the sidebar instead of a text going out.
            """)

    elif option == "Image":

        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        use_tta = st.checkbox(
            "Boost detection on hard/crowded images (slower)",
            value=False,
            help="Runs the image through the model multiple times with slight variations and "
                 "merges the results. Can catch a few extra weak detections in dense/small-"
                 "figure scenes, at the cost of taking noticeably longer per image. Only "
                 "available here — too slow for live video/webcam."
        )
        if uploaded:
            img = Image.open(uploaded).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)
            results = model.predict(
                source=img, conf=conf_threshold, imgsz=inference_size, iou=nms_iou,
                augment=use_tta, verbose=False,
            )
            # unique tag per upload so people in different photos never share a Worker ID
            frame_tag = f"img_{uploaded.name}_{uploaded.size}"
            fw, fh = img.size  # PIL: (width, height)
            worker_annotations = process_violations(results, "Image", frame_tag, fw, fh)
            # Re-draw the emergency/alert banners now that this image's detections have been
            # processed — they were rendered once at the top of the page, before this upload
            # even ran, so without this refresh any emergency triggered by this image stays
            # invisible in session state instead of showing up on screen.
            with emergency_slot.container():
                render_emergency()
            with alert_slot.container():
                render_alerts()
            annotated = results[0].plot()
            annotated = draw_worker_overlay(annotated, worker_annotations)
            st.image(annotated, caption="Detected Output — green = compliant, red = missing PPE",
                      channels="BGR", use_container_width=True)

    elif option == "Video":
        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded.read())
                video_path = tmp.name

            if st.button("Start Video Detection"):
                reset_session_identities()  # fresh session: don't inherit IDs from a past run
                cap = cv2.VideoCapture(video_path)
                frame_window = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    results = model.track(frame, conf=conf_threshold, imgsz=inference_size, iou=nms_iou, persist=True, tracker="botsort.yaml", verbose=False)
                    fh, fw = frame.shape[:2]  # cv2: (height, width, channels)
                    worker_annotations = process_violations(results, "Video", "video", fw, fh)
                    with emergency_slot.container():
                        render_emergency()
                    with alert_slot.container():
                        render_alerts()
                    annotated_frame = results[0].plot()
                    annotated_frame = draw_worker_overlay(annotated_frame, worker_annotations)
                    frame_window.image(annotated_frame, channels="BGR", use_container_width=True)
                    time.sleep(0.03)

                cap.release()
                st.success("Video finished.")

    elif option == "Webcam":
        start = st.button("Start Webcam")
        stop = st.button("Stop Webcam")
        frame_window = st.empty()

        if start:
            reset_session_identities()  # fresh session: don't inherit IDs from a past run
            cap = cv2.VideoCapture(0)

            while cap.isOpened() and not stop:
                ret, frame = cap.read()
                if not ret:
                    st.error("Cannot access webcam.")
                    break

                results = model.track(frame, conf=conf_threshold, imgsz=inference_size, iou=nms_iou, persist=True, tracker="botsort.yaml", verbose=False)
                fh, fw = frame.shape[:2]  # cv2: (height, width, channels)
                worker_annotations = process_violations(results, "Webcam", "webcam", fw, fh)
                with emergency_slot.container():
                    render_emergency()
                with alert_slot.container():
                    render_alerts()
                annotated_frame = results[0].plot()
                annotated_frame = draw_worker_overlay(annotated_frame, worker_annotations)
                frame_window.image(annotated_frame, channels="BGR", use_container_width=True)
                time.sleep(0.03)

            cap.release()
            st.success("Webcam stopped.")