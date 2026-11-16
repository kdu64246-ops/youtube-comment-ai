import os
import re
import json
import hashlib
from collections import Counter
from functools import lru_cache
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# ============================ 기본 설정 ============================
MAX_COMMENTS = 20000
DATA_DIR = "data"

KOREAN_ONLY = True
MIN_KOREAN_CHARS = 1

EXCLUDE_DAYS = 3
SPIKE_MULT = 5
MIN_SPIKE = 20

TFIDF_NEAR_DUP_SIM = 0.90
TFIDF_SAMPLE = 3000

USE_SEMANTIC_AI = True
EMBED_MODEL = "jhgan/ko-sroberta-multitask"
SEMANTIC_SAMPLE = 2500
SEMANTIC_SIM = 0.85
CLUSTER_EPS = 0.18
CLUSTER_MIN_SAMPLES = 5

USE_SENTIMENT = True
SENTIMENT_MODEL = "cringepnh/koelectra-korean-sentiment"
SENT_SAMPLE = 3000

CACHE_VERSION = "web_v1"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(url_or_id):
    """유튜브 주소 또는 11자리 영상 ID에서 영상 ID를 추출."""
    s = str(url_or_id).strip()
    if _ID_RE.match(s):
        return s

    if not re.match(r"^https?://", s, re.I):
        s = "https://" + s

    u = urlparse(s)
    host = (u.hostname or "").lower()
    vid = None

    if host.endswith("youtu.be"):
        vid = u.path.lstrip("/").split("/")[0]
    elif host.endswith("youtube.com") or host.endswith("youtube-nocookie.com"):
        q = parse_qs(u.query)
        if q.get("v"):
            vid = q["v"][0]
        else:
            m = re.search(r"/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})", u.path)
            if m:
                vid = m.group(1)

    if vid and _ID_RE.match(vid[:11]):
        return vid[:11]

    raise ValueError("유튜브 영상 주소로 인식할 수 없습니다.")


def normalize(text):
    t = re.sub(r"[^0-9a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ ]", " ", str(text).lower())
    return re.sub(r"\s+", " ", t).strip()


def korean_char_count(text):
    return len(re.findall(r"[가-힣]", str(text)))


def filter_analysis_comments(df):
    out = df.copy()
    out["text"] = out["text"].fillna("").astype(str)
    out["norm_text"] = out["text"].map(normalize)
    out = out[out["norm_text"].str.len() >= 2].copy()

    if KOREAN_ONLY:
        out = out[out["text"].map(korean_char_count) >= MIN_KOREAN_CHARS].copy()

    return out.reset_index(drop=True)


def get_youtube(api_key):
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def fetch_video_info(youtube, video_id):
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{video_id}_{CACHE_VERSION}_info.json")

    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            info = json.load(f)
    else:
        res = youtube.videos().list(
            part="snippet,statistics",
            id=video_id
        ).execute()

        if not res.get("items"):
            raise ValueError("영상을 찾을 수 없습니다. 비공개·삭제 영상 또는 잘못된 주소일 수 있습니다.")

        it = res["items"][0]
        info = {
            "title": it["snippet"].get("title", ""),
            "upload": it["snippet"]["publishedAt"],
            "view_count": int(it["statistics"].get("viewCount", 0)),
            "reported_comments": int(it["statistics"].get("commentCount", 0)),
        }

        with open(cache, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)

    info["upload"] = pd.to_datetime(info["upload"], utc=True)
    return info


def fetch_comments(youtube, video_id):
    os.makedirs(DATA_DIR, exist_ok=True)
    cache = os.path.join(DATA_DIR, f"{video_id}_{CACHE_VERSION}.csv")

    if os.path.exists(cache):
        df = pd.read_csv(
            cache,
            dtype={"author_hash": str, "text": str},
            keep_default_na=False
        )
        df["published"] = pd.to_datetime(df["published"], utc=True)
        return df

    rows = []
    token = None

    while len(rows) < MAX_COMMENTS:
        res = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=token,
            textFormat="plainText",
            order="time",
        ).execute()

        for item in res.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            author_key = (
                (s.get("authorChannelId") or {}).get("value")
                or s.get("authorDisplayName", "")
            )

            rows.append({
                "author_hash": hashlib.sha256(
                    author_key.encode("utf-8")
                ).hexdigest()[:12],
                "text": s.get("textDisplay", ""),
                "published": s["publishedAt"],
                "likes": s.get("likeCount", 0),
            })

            if len(rows) >= MAX_COMMENTS:
                break

        token = res.get("nextPageToken")
        if not token:
            break

    df = pd.DataFrame(
        rows,
        columns=["author_hash", "text", "published", "likes"]
    )

    if not df.empty:
        df["published"] = pd.to_datetime(df["published"], utc=True)
        df.to_csv(cache, index=False, encoding="utf-8-sig")

    return df


def time_metrics(df, upload):
    hours = (df["published"] - upload).dt.total_seconds() / 3600
    share_48h = float((hours <= 48).mean()) if len(df) else np.nan

    daily = df.set_index("published").sort_index().resample("1D").size()
    prev = daily.shift(1).rolling(7, min_periods=3).median()
    threshold = np.maximum(MIN_SPIKE, prev * SPIKE_MULT)

    after = daily.index >= (
        upload + pd.Timedelta(days=EXCLUDE_DAYS)
    ).floor("D")

    respikes = daily[after & (daily > threshold)]

    return {
        "share_48h": share_48h,
        "daily": daily,
        "threshold": threshold.where(after),
        "respikes": respikes,
    }


def tfidf_duplicate_metrics(df):
    valid = df["norm_text"]
    counts = valid.value_counts()

    exact_share = float(valid.map(counts).gt(1).mean()) if len(valid) else 0.0
    top_phrases = [
        {"text": t[:80], "count": int(c)}
        for t, c in counts.head(5).items()
        if c > 1
    ]

    sample = valid.sample(min(TFIDF_SAMPLE, len(valid)), random_state=0)
    near_share = 0.0

    if len(sample) >= 2:
        tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 3)
        ).fit_transform(sample)

        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0)
        near_share = float(
            (sim.max(axis=1) >= TFIDF_NEAR_DUP_SIM).mean()
        )

    return {
        "exact_share": exact_share,
        "tfidf_near_share": near_share,
        "top_phrases": top_phrases,
    }


@lru_cache(maxsize=1)
def _load_embedder():
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(EMBED_MODEL, device=device)


def semantic_ai_metrics(df):
    empty = {
        "semantic_near_share": np.nan,
        "semantic_cluster_count": 0,
        "largest_cluster_share": 0.0,
        "clustered_share": 0.0,
        "top_clusters": [],
    }

    if not USE_SEMANTIC_AI or len(df) < 2:
        return empty

    model = _load_embedder()

    sample = df.sample(
        min(SEMANTIC_SAMPLE, len(df)),
        random_state=42
    ).copy()

    texts = sample["text"].astype(str).str[:300].tolist()

    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    sim = emb @ emb.T
    np.fill_diagonal(sim, -1.0)

    max_sim = sim.max(axis=1)
    semantic_near_share = float((max_sim >= SEMANTIC_SIM).mean())

    labels = DBSCAN(
        eps=CLUSTER_EPS,
        min_samples=CLUSTER_MIN_SAMPLES,
        metric="cosine",
        n_jobs=-1,
    ).fit_predict(emb)

    clustered = labels != -1
    cluster_ids = sorted(set(labels) - {-1})
    cluster_sizes = Counter(labels[clustered]) if clustered.any() else Counter()

    clustered_share = float(clustered.mean())
    largest_cluster_share = (
        max(cluster_sizes.values()) / len(sample)
        if cluster_sizes else 0.0
    )

    top_clusters = []

    for cid, size in cluster_sizes.most_common(5):
        idxs = np.where(labels == cid)[0]

        centroid = emb[idxs].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        center_sim = emb[idxs] @ centroid
        order = idxs[np.argsort(center_sim)[::-1][:5]]

        examples = [
            str(sample.iloc[i]["text"])[:180]
            for i in order
        ]

        top_clusters.append({
            "cluster_id": int(cid),
            "size": int(size),
            "examples": examples,
        })

    return {
        "semantic_near_share": semantic_near_share,
        "semantic_cluster_count": len(cluster_ids),
        "largest_cluster_share": float(largest_cluster_share),
        "clustered_share": clustered_share,
        "top_clusters": top_clusters,
    }


def author_metrics(df):
    if df.empty:
        return {
            "heavy_author_share": 0.0,
            "top_author_share": 0.0
        }

    per_author = df["author_hash"].value_counts()
    heavy = per_author[per_author >= 3]

    return {
        "heavy_author_share": float(heavy.sum() / len(df)),
        "top_author_share": float(per_author.iloc[0] / len(df)),
    }


def _positive_label(label):
    l = str(label).strip().lower()

    if "positive" in l or "pos" in l or "긍정" in l:
        return 1
    if "negative" in l or "neg" in l or "부정" in l:
        return 0
    if l in {"label_1", "1"}:
        return 1
    if l in {"label_0", "0"}:
        return 0

    return None


@lru_cache(maxsize=1)
def _load_sentiment_model():
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=SENTIMENT_MODEL,
        device=device
    )


def sentiment_metrics(df):
    if not USE_SENTIMENT or df.empty:
        return {"positive": np.nan}

    clf = _load_sentiment_model()

    s = df.sample(min(SENT_SAMPLE, len(df)), random_state=0)

    res = clf(
        s["text"].astype(str).str[:200].tolist(),
        batch_size=64,
        truncation=True,
    )

    parsed = [_positive_label(r["label"]) for r in res]
    parsed = [x for x in parsed if x is not None]

    return {
        "positive": float(np.mean(parsed)) if parsed else np.nan
    }


def simulation_signals(metrics):
    """
    발표/시뮬레이션용 탐색 신호.
    실제 댓글부대 존재를 증명하는 기준이 아님.
    """
    signals = []

    if metrics["respike_days"] >= 1:
        signals.append("업로드 이후 비정상적인 댓글 재급증")

    if metrics["tfidf_near_share"] >= 0.10:
        signals.append("문자 형태가 매우 비슷한 댓글 비율이 높음")

    if (
        not np.isnan(metrics["semantic_near_share"])
        and metrics["semantic_near_share"] >= 0.15
    ):
        signals.append("AI가 의미가 매우 비슷한 댓글을 다수 탐지")

    if metrics["largest_cluster_share"] >= 0.05:
        signals.append("특정 AI 의미 군집에 댓글이 집중")

    if metrics["heavy_author_share"] >= 0.10:
        signals.append("반복 작성자의 댓글 비중이 높음")

    # 확정 판정이 아니라, 사람이 다시 확인할 필요가 있는 정도만 표시
    if len(signals) >= 3:
        level = "여러 이상 신호 발견 — 추가 확인 필요"
    elif len(signals) >= 1:
        level = "일부 이상 신호 발견"
    else:
        level = "두드러진 이상 신호가 적음"

    return level, signals


def analyze_video(url, api_key):
    video_id = extract_video_id(url)
    youtube = get_youtube(api_key)

    info = fetch_video_info(youtube, video_id)
    raw_df = fetch_comments(youtube, video_id)

    if raw_df.empty:
        raise ValueError("수집 가능한 댓글이 없습니다.")

    df = filter_analysis_comments(raw_df)

    if df.empty:
        raise ValueError("분석 가능한 한국어 댓글이 없습니다.")

    time = time_metrics(df, info["upload"])
    dup = tfidf_duplicate_metrics(df)
    sem = semantic_ai_metrics(df)
    auth = author_metrics(df)
    sent = sentiment_metrics(df)

    metrics = {
        "share_48h": time["share_48h"],
        "respike_days": len(time["respikes"]),
        "exact_share": dup["exact_share"],
        "tfidf_near_share": dup["tfidf_near_share"],
        "semantic_near_share": sem["semantic_near_share"],
        "semantic_cluster_count": sem["semantic_cluster_count"],
        "clustered_share": sem["clustered_share"],
        "largest_cluster_share": sem["largest_cluster_share"],
        "heavy_author_share": auth["heavy_author_share"],
        "top_author_share": auth["top_author_share"],
        "positive": sent["positive"],
    }

    level, signals = simulation_signals(metrics)

    daily_df = pd.DataFrame({
        "날짜": time["daily"].index,
        "댓글 수": time["daily"].values,
    })

    return {
        "video_id": video_id,
        "title": info["title"],
        "upload": info["upload"],
        "view_count": info["view_count"],
        "reported_comments": info["reported_comments"],
        "collected_comments": len(raw_df),
        "analyzed_comments": len(df),
        "metrics": metrics,
        "top_phrases": dup["top_phrases"],
        "top_clusters": sem["top_clusters"],
        "daily": daily_df,
        "level": level,
        "signals": signals,
    }
