import os
import re
import json
import gc
import hashlib
import pickle
from collections import Counter
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

# ============================ KOTE 44감정 분석 ============================
# KOTE 공식 GitHub에서 배포한 Hugging Face Trainer 버전.
# KcELECTRA-base를 KOTE 44개 멀티라벨 감정으로 파인튜닝한 체크포인트.
KOTE_MODEL = "searle-j/kote_for_easygoing_people"

# 공개 모델 예제에서 score > 0.4를 감정 라벨 채택 기준으로 사용.
KOTE_THRESHOLD = 0.40

# 웹 배포 환경의 메모리/속도를 고려해 표본 분석.
# 분석 화면에 실제 사용 표본 수를 함께 표시한다.
KOTE_SAMPLE = 1500
KOTE_BATCH_SIZE = 32

CACHE_VERSION = "web_kote44_v1"

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# KOTE의 44개 라벨. 모델 config의 id2label을 우선 사용하고,
# 안전한 폴백 및 표시 순서 확인용으로 명시한다.
KOTE_LABELS = [
    "불평/불만", "환영/호의", "감동/감탄", "지긋지긋", "고마움", "슬픔",
    "화남/분노", "존경", "기대감", "우쭐댐/무시함", "안타까움/실망", "비장함",
    "의심/불신", "뿌듯함", "편안/쾌적", "신기함/관심", "아껴주는", "부끄러움",
    "공포/무서움", "절망", "한심함", "역겨움/징그러움", "짜증", "어이없음",
    "없음", "패배/자기혐오", "귀찮음", "힘듦/지침", "즐거움/신남", "깨달음",
    "죄책감", "증오/혐오", "흐뭇함(귀여움/예쁨)", "당황/난처", "경악",
    "부담/안_내킴", "서러움", "재미없음", "불쌍함/연민", "놀람", "행복",
    "불안/걱정", "기쁨", "안심/신뢰",
]

# KOTE 논문 Figure 2의 valence 색상 분류를 프로젝트의 3분류 요약에 사용.
# '없음(NO EMOTION)'은 중립에 포함한다.
NEGATIVE_EMOTIONS = {
    "지긋지긋", "경악", "역겨움/징그러움", "증오/혐오", "화남/분노", "짜증",
    "불평/불만", "어이없음", "한심함", "의심/불신", "안타까움/실망",
    "당황/난처", "부끄러움", "죄책감", "패배/자기혐오", "공포/무서움",
    "불안/걱정", "불쌍함/연민", "슬픔", "서러움", "절망", "힘듦/지침",
    "귀찮음", "부담/안_내킴", "재미없음",
}

NEUTRAL_EMOTIONS = {
    "우쭐댐/무시함", "비장함", "없음", "깨달음", "놀람", "존경", "신기함/관심",
}

POSITIVE_EMOTIONS = {
    "기대감", "환영/호의", "아껴주는", "흐뭇함(귀여움/예쁨)", "즐거움/신남",
    "기쁨", "행복", "감동/감탄", "뿌듯함", "고마움", "안심/신뢰", "편안/쾌적",
}


def extract_video_id(url_or_id):
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

        # 전체 3000x3000 유사도 행렬을 한 번에 만들지 않고
        # 작은 블록으로 계산하여 Streamlit Cloud 메모리 사용을 줄인다.
        n = tfidf.shape[0]
        block = 256
        max_sim = np.zeros(n, dtype=np.float32)

        for start in range(0, n, block):
            end = min(start + block, n)
            sim_block = cosine_similarity(tfidf[start:end], tfidf)

            # 자기 자신과의 유사도는 제외
            rows = np.arange(end - start)
            cols = np.arange(start, end)
            sim_block[rows, cols] = 0.0

            max_sim[start:end] = sim_block.max(axis=1).astype(np.float32)

        near_share = float((max_sim >= TFIDF_NEAR_DUP_SIM).mean())

        del tfidf, max_sim
        gc.collect()

    return {
        "exact_share": exact_share,
        "tfidf_near_share": near_share,
        "top_phrases": top_phrases,
    }


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

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBED_MODEL, device=device)

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

    # 전체 NxN 행렬을 메모리에 유지하지 않고 블록 단위로
    # 각 댓글의 최대 의미 유사도만 계산한다.
    n = emb.shape[0]
    block = 256
    max_sim = np.full(n, -1.0, dtype=np.float32)

    for start in range(0, n, block):
        end = min(start + block, n)
        sim_block = emb[start:end] @ emb.T

        rows = np.arange(end - start)
        cols = np.arange(start, end)
        sim_block[rows, cols] = -1.0

        max_sim[start:end] = sim_block.max(axis=1)

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

    # 다음 KOTE 모델을 위해 임베딩 모델 메모리 해제
    del model, emb, max_sim
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def _normalize_kote_label(label):
    """모델/논문 표기 차이를 통일."""
    label = str(label)
    aliases = {
        "NO EMOTION": "없음",
        "정서 없음": "없음",
        "부담/안 내킴": "부담/안_내킴",
    }
    return aliases.get(label, label)


def _valence_group(label):
    label = _normalize_kote_label(label)
    if label in NEGATIVE_EMOTIONS:
        return "부정"
    if label in POSITIVE_EMOTIONS:
        return "긍정"
    return "중립"


def kote_emotion_metrics(df):
    """
    KOTE 44개 감정을 멀티라벨로 분석한다.
    - sigmoid 확률 사용
    - KOTE_THRESHOLD 이상인 감정을 해당 댓글의 감정으로 채택
    - 세부 감정 비율은 멀티라벨이므로 합계가 100%를 넘을 수 있음
    - 긍/부정/중립 요약은 댓글마다 각 그룹의 '가장 강한 감정 확률'을 비교해 1개로 분류
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    if df.empty:
        return {
            "sample_size": 0,
            "threshold": KOTE_THRESHOLD,
            "emotion_distribution": [],
            "valence_distribution": {"긍정": 0.0, "부정": 0.0, "중립": 0.0},
            "top_emotion_examples": {},
        }

    sample = df.sample(
        min(KOTE_SAMPLE, len(df)),
        random_state=0
    ).copy()

    texts = sample["text"].astype(str).str[:500].tolist()

    tokenizer = AutoTokenizer.from_pretrained(KOTE_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(KOTE_MODEL)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 모델 config의 실제 라벨 순서를 사용
    id2label = {
        int(k): _normalize_kote_label(v)
        for k, v in model.config.id2label.items()
    }
    labels = [id2label[i] for i in range(len(id2label))]

    if len(labels) != 44:
        raise ValueError(
            f"KOTE 모델 라벨 수가 44개가 아닙니다: {len(labels)}개"
        )

    all_probs = []

    with torch.no_grad():
        for start in range(0, len(texts), KOTE_BATCH_SIZE):
            batch_texts = texts[start:start + KOTE_BATCH_SIZE]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = model(**enc).logits

            # KOTE는 multi-label이므로 softmax가 아니라 sigmoid
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)

    probs = np.vstack(all_probs)

    # -------- 44개 세부 감정 집계 --------
    detected = probs >= KOTE_THRESHOLD

    emotion_distribution = []
    for i, label in enumerate(labels):
        count = int(detected[:, i].sum())
        emotion_distribution.append({
            "감정": label,
            "댓글 수": count,
            "비율": float(count / len(sample)),
            "평균 확률": float(probs[:, i].mean()),
            "분류": _valence_group(label),
        })

    emotion_distribution.sort(
        key=lambda x: (x["댓글 수"], x["평균 확률"]),
        reverse=True
    )

    # -------- 댓글별 긍정/부정/중립 1개 요약 --------
    label_to_idx = {label: i for i, label in enumerate(labels)}
    neg_idx = [label_to_idx[x] for x in NEGATIVE_EMOTIONS if x in label_to_idx]
    pos_idx = [label_to_idx[x] for x in POSITIVE_EMOTIONS if x in label_to_idx]
    neu_idx = [label_to_idx[x] for x in NEUTRAL_EMOTIONS if x in label_to_idx]

    # 각 그룹에서 가장 강한 감정의 확률을 그 그룹 점수로 사용
    neg_score = probs[:, neg_idx].max(axis=1) if neg_idx else np.zeros(len(sample))
    pos_score = probs[:, pos_idx].max(axis=1) if pos_idx else np.zeros(len(sample))
    neu_score = probs[:, neu_idx].max(axis=1) if neu_idx else np.zeros(len(sample))

    group_scores = np.stack([pos_score, neg_score, neu_score], axis=1)
    group_names = np.array(["긍정", "부정", "중립"])

    winners = group_scores.argmax(axis=1)
    max_scores = group_scores.max(axis=1)

    # 어느 그룹도 임계값을 넘지 못하면 중립으로 처리
    valence_labels = group_names[winners]
    valence_labels[max_scores < KOTE_THRESHOLD] = "중립"

    valence_counts = Counter(valence_labels.tolist())
    valence_distribution = {
        name: float(valence_counts.get(name, 0) / len(sample))
        for name in ["긍정", "부정", "중립"]
    }

    # -------- 상위 감정별 대표 댓글 --------
    top_emotion_examples = {}
    for item in emotion_distribution[:8]:
        label = item["감정"]
        i = label_to_idx[label]

        # 해당 감정이 임계값을 넘은 댓글 중 확률 상위 3개
        idxs = np.where(probs[:, i] >= KOTE_THRESHOLD)[0]
        if len(idxs) == 0:
            continue

        order = idxs[np.argsort(probs[idxs, i])[::-1][:3]]
        top_emotion_examples[label] = [
            {
                "text": str(sample.iloc[j]["text"])[:220],
                "score": float(probs[j, i]),
            }
            for j in order
        ]

    # 메모리 해제
    del model, tokenizer, probs, all_probs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "sample_size": len(sample),
        "threshold": KOTE_THRESHOLD,
        "emotion_distribution": emotion_distribution,
        "valence_distribution": valence_distribution,
        "top_emotion_examples": top_emotion_examples,
    }



def organizational_score(metrics):
    """
    과제용 '조직적 댓글 활동' 판정 점수 (총 120점).

    5개 핵심 지표를 각각 0/10/20점으로 평가하고,
    시간·내용·작성자 세 영역이 동시에 이상하면 복합 신호 20점을 추가한다.

    ※ 이 점수는 본 프로젝트에서 정의한 운영 기준이다.
    """

    item_scores = {}
    reasons = []

    # 1) 댓글 재급증: 0 / 10 / 20
    respike_days = int(metrics["respike_days"])
    if respike_days >= 2:
        item_scores["댓글 재급증"] = 20
        reasons.append(f"댓글 재급증이 {respike_days}일 발생")
    elif respike_days == 1:
        item_scores["댓글 재급증"] = 10
        reasons.append("댓글 재급증이 1일 발생")
    else:
        item_scores["댓글 재급증"] = 0

    # 2) AI 의미 유사 댓글: 0 / 10 / 20
    sem = metrics["semantic_near_share"]
    if not np.isnan(sem) and sem >= 0.20:
        item_scores["AI 의미 유사 댓글"] = 20
        reasons.append(f"AI 의미 유사 댓글 비율이 {sem:.1%}")
    elif not np.isnan(sem) and sem >= 0.10:
        item_scores["AI 의미 유사 댓글"] = 10
        reasons.append(f"AI 의미 유사 댓글 비율이 {sem:.1%}")
    else:
        item_scores["AI 의미 유사 댓글"] = 0

    # 3) 최대 AI 군집 비율: 0 / 10 / 20
    cluster = metrics["largest_cluster_share"]
    if cluster >= 0.07:
        item_scores["최대 AI 군집"] = 20
        reasons.append(f"가장 큰 AI 의미 군집이 전체 표본의 {cluster:.1%}")
    elif cluster >= 0.03:
        item_scores["최대 AI 군집"] = 10
        reasons.append(f"가장 큰 AI 의미 군집이 전체 표본의 {cluster:.1%}")
    else:
        item_scores["최대 AI 군집"] = 0

    # 4) 반복 작성자 댓글: 0 / 10 / 20
    heavy = metrics["heavy_author_share"]
    if heavy >= 0.10:
        item_scores["반복 작성자"] = 20
        reasons.append(f"반복 작성자 댓글 비율이 {heavy:.1%}")
    elif heavy >= 0.05:
        item_scores["반복 작성자"] = 10
        reasons.append(f"반복 작성자 댓글 비율이 {heavy:.1%}")
    else:
        item_scores["반복 작성자"] = 0

    # 5) TF-IDF 유사 댓글: 0 / 10 / 20
    tfidf = metrics["tfidf_near_share"]
    if tfidf >= 0.08:
        item_scores["TF-IDF 유사 댓글"] = 20
        reasons.append(f"문자 형태가 매우 비슷한 댓글 비율이 {tfidf:.1%}")
    elif tfidf >= 0.03:
        item_scores["TF-IDF 유사 댓글"] = 10
        reasons.append(f"문자 형태가 매우 비슷한 댓글 비율이 {tfidf:.1%}")
    else:
        item_scores["TF-IDF 유사 댓글"] = 0

    # 6) 복합 신호 20점
    # 시간 / 내용 / 작성자 행동 세 영역이 동시에 기준을 넘으면 추가점수
    time_abnormal = respike_days >= 1
    content_abnormal = (
        (not np.isnan(sem) and sem >= 0.15)
        or cluster >= 0.05
        or tfidf >= 0.08
    )
    author_abnormal = heavy >= 0.10

    if time_abnormal and content_abnormal and author_abnormal:
        item_scores["복합 신호"] = 20
        reasons.append("시간·댓글 내용·작성자 행동의 이상 신호가 동시에 발생")
    else:
        item_scores["복합 신호"] = 0

    total = int(sum(item_scores.values()))

    if total >= 70:
        verdict = "조직적 댓글 활동 강하게 의심"
        grade = "고위험"
    elif total >= 50:
        verdict = "조직적 댓글 활동 의심"
        grade = "의심"
    elif total >= 30:
        verdict = "일부 비정상 댓글 패턴 발견"
        grade = "주의"
    else:
        verdict = "정상적인 댓글 패턴"
        grade = "정상"

    return {
        "score": total,
        "max_score": 120,
        "grade": grade,
        "verdict": verdict,
        "item_scores": item_scores,
        "reasons": reasons,
        "domain_flags": {
            "시간": time_abnormal,
            "내용": content_abnormal,
            "작성자": author_abnormal,
        },
    }



def save_single_checkpoint(path, result):
    """단일 영상 결과를 세션과 별개로 서버 파일에 저장."""
    if not path:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_single_checkpoint(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def analyze_video(url, api_key, checkpoint_path=None, progress_callback=None):
    """
    안정형 단일영상 분석.

    핵심 120점 결과를 KOTE보다 먼저 완성하고 체크포인트에 저장한다.
    따라서 KOTE 감정 모델에서 오류가 나도 점수 결과와 다운로드 자료는 유지된다.
    """
    def report(step, message):
        if progress_callback:
            progress_callback(step, message)

    report(0.05, "영상 정보와 댓글을 수집하는 중")
    video_id = extract_video_id(url)
    youtube = get_youtube(api_key)

    info = fetch_video_info(youtube, video_id)
    raw_df = fetch_comments(youtube, video_id)

    if raw_df.empty:
        raise ValueError("수집 가능한 댓글이 없습니다.")

    df = filter_analysis_comments(raw_df)
    if df.empty:
        raise ValueError("분석 가능한 한국어 댓글이 없습니다.")

    report(0.25, "시간 패턴·TF-IDF·작성자 행동을 분석하는 중")
    time = time_metrics(df, info["upload"])
    dup = tfidf_duplicate_metrics(df)
    auth = author_metrics(df)

    report(0.45, "Ko-SRoBERTa로 의미 유사도와 의미 군집을 분석하는 중")
    sem = semantic_ai_metrics(df)

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
    }

    report(0.68, "120점 조직적 댓글 패턴 점수를 계산하는 중")
    score_result = organizational_score(metrics)

    daily_df = pd.DataFrame({
        "날짜": time["daily"].index,
        "댓글 수": time["daily"].values,
    })

    # KOTE 전에도 완전한 핵심 결과를 만든다.
    result = {
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
        "score_result": score_result,

        # KOTE는 아래에서 채운다. 실패해도 핵심 결과는 유지.
        "emotion_sample_size": 0,
        "emotion_threshold": KOTE_THRESHOLD,
        "emotion_distribution": [],
        "valence_distribution": {"긍정": 0.0, "부정": 0.0, "중립": 0.0},
        "top_emotion_examples": {},
        "emotion_error": None,
    }

    # 핵심 결과를 먼저 저장
    save_single_checkpoint(checkpoint_path, result)

    report(0.76, "핵심 점수 저장 완료 · KOTE 44개 감정을 분석하는 중")

    try:
        emotion = kote_emotion_metrics(df)
        result.update({
            "emotion_sample_size": emotion["sample_size"],
            "emotion_threshold": emotion["threshold"],
            "emotion_distribution": emotion["emotion_distribution"],
            "valence_distribution": emotion["valence_distribution"],
            "top_emotion_examples": emotion["top_emotion_examples"],
            "emotion_error": None,
        })
    except Exception as e:
        # 감정 분석은 점수에 직접 들어가지 않으므로 실패해도 결과 반환
        result["emotion_error"] = str(e)

    save_single_checkpoint(checkpoint_path, result)
    report(1.0, "분석 완료")

    return result
