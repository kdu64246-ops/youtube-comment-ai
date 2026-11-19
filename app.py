import os
import io
import zipfile
import re
import numpy as np
import pandas as pd
import streamlit as st

from analyzer import analyze_videos_batch

def _pct(value):
    """0~1 비율을 발표용 % 문자열로 변환."""
    try:
        if value is None or np.isnan(value):
            return "분석값 없음"
    except TypeError:
        pass
    return f"{float(value) * 100:.1f}%"


def _csv_bytes(df):
    """Excel에서 한글이 깨지지 않도록 UTF-8 BOM을 포함한 CSV 생성."""
    return df.to_csv(index=False).encode("utf-8-sig")


def build_ppt_result_zip(result):
    """
    한 번의 분석 결과를 발표자료 제작에 바로 사용할 수 있는 ZIP으로 묶는다.
    포함 파일:
      01_분석요약.csv
      02_점수산정.csv
      03_KOTE_44감정.csv
      04_긍정부정중립.csv
      05_일별댓글수.csv
      06_반복문구.csv
      07_AI의미군집.csv
      08_감정별대표댓글.csv
      PPT_발표요약.txt
    """
    m = result["metrics"]
    score = result["score_result"]
    vd = result["valence_distribution"]

    # ---------------- 01. 분석 요약 ----------------
    summary_rows = [
        ["영상", "제목", result["title"], ""],
        ["영상", "영상 ID", result["video_id"], ""],
        ["영상", "업로드 날짜", str(result["upload"]), ""],
        ["영상", "조회수", result["view_count"], "회"],
        ["데이터", "YouTube 표시 댓글 수", result["reported_comments"], "개"],
        ["데이터", "실제 수집 댓글 수", result["collected_comments"], "개"],
        ["데이터", "한국어 분석 댓글 수", result["analyzed_comments"], "개"],
        ["분석", "업로드 후 48시간 댓글 비율", _pct(m["share_48h"]), "보조 지표"],
        ["분석", "댓글 재급증", m["respike_days"], "일"],
        ["분석", "TF-IDF 유사 댓글", _pct(m["tfidf_near_share"]), "점수 반영"],
        ["AI 분석", "Ko-SRoBERTa 의미 유사 댓글", _pct(m["semantic_near_share"]), "점수 반영"],
        ["AI 분석", "AI 의미 군집 개수", m["semantic_cluster_count"], "개"],
        ["AI 분석", "가장 큰 AI 의미 군집", _pct(m["largest_cluster_share"]), "점수 반영"],
        ["분석", "반복 작성자 댓글", _pct(m["heavy_author_share"]), "점수 반영"],
        ["AI 감정", "긍정", _pct(vd["긍정"]), "KOTE 보조 분석"],
        ["AI 감정", "부정", _pct(vd["부정"]), "KOTE 보조 분석"],
        ["AI 감정", "중립", _pct(vd["중립"]), "KOTE 보조 분석"],
        ["최종 판정", "조직적 활동 점수", score["score"], f"/ {score['max_score']}"],
        ["최종 판정", "등급", score["grade"], ""],
        ["최종 판정", "판정", score["verdict"], ""],
    ]
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["구분", "항목", "값", "단위/설명"]
    )

    # ---------------- 02. 점수 산정 ----------------
    score_df = pd.DataFrame([
        {
            "점수 항목": item,
            "획득 점수": value,
            "최대 점수": 20,
        }
        for item, value in score["item_scores"].items()
    ])
    score_df.loc[len(score_df)] = {
        "점수 항목": "총점",
        "획득 점수": score["score"],
        "최대 점수": score["max_score"],
    }

    # ---------------- 03. KOTE 44감정 ----------------
    emotion_df = pd.DataFrame(result["emotion_distribution"])
    if not emotion_df.empty:
        emotion_df = emotion_df.rename(columns={
            "감정": "세부 감정",
            "분류": "긍정/부정/중립",
            "댓글 수": "검출 댓글 수",
            "비율": "검출 비율",
            "평균 확률": "평균 확률",
        })
        emotion_df["검출 비율(%)"] = (emotion_df["검출 비율"] * 100).round(2)
        emotion_df["평균 확률"] = emotion_df["평균 확률"].round(4)
        emotion_df = emotion_df[
            ["세부 감정", "긍정/부정/중립", "검출 댓글 수", "검출 비율(%)", "평균 확률"]
        ]

    # ---------------- 04. 긍정/부정/중립 ----------------
    valence_df = pd.DataFrame({
        "분류": ["긍정", "부정", "중립"],
        "비율(%)": [
            round(vd["긍정"] * 100, 2),
            round(vd["부정"] * 100, 2),
            round(vd["중립"] * 100, 2),
        ]
    })

    # ---------------- 05. 일별 댓글 수 ----------------
    daily_df = result["daily"].copy()
    if "날짜" in daily_df.columns:
        daily_df["날짜"] = daily_df["날짜"].astype(str)

    # ---------------- 06. 반복 문구 ----------------
    phrase_df = pd.DataFrame(result["top_phrases"])
    if not phrase_df.empty:
        phrase_df = phrase_df.rename(columns={"text": "반복 댓글", "count": "반복 횟수"})

    # ---------------- 07. AI 의미 군집 ----------------
    cluster_rows = []
    for cluster in result["top_clusters"]:
        for rank, text in enumerate(cluster.get("examples", []), 1):
            cluster_rows.append({
                "군집 ID": cluster["cluster_id"],
                "군집 크기": cluster["size"],
                "대표 댓글 순위": rank,
                "대표 댓글": text,
            })
    cluster_df = pd.DataFrame(
        cluster_rows,
        columns=["군집 ID", "군집 크기", "대표 댓글 순위", "대표 댓글"]
    )

    # ---------------- 08. 감정별 대표 댓글 ----------------
    emotion_example_rows = []
    for emotion, rows in result["top_emotion_examples"].items():
        for rank, row in enumerate(rows, 1):
            emotion_example_rows.append({
                "감정": emotion,
                "순위": rank,
                "AI 확률(%)": round(float(row["score"]) * 100, 2),
                "대표 댓글": row["text"],
            })
    emotion_example_df = pd.DataFrame(
        emotion_example_rows,
        columns=["감정", "순위", "AI 확률(%)", "대표 댓글"]
    )

    # ---------------- PPT에 복붙하기 쉬운 요약 TXT ----------------
    item_scores_text = "\n".join(
        f"- {item}: {value}/20점"
        for item, value in score["item_scores"].items()
    )
    reasons_text = "\n".join(
        f"- {reason}" for reason in score.get("reasons", [])
    ) or "- 두드러진 판정 근거 없음"

    top_emotions = []
    for row in result["emotion_distribution"][:5]:
        top_emotions.append(
            f"- {row['감정']}: {row['비율'] * 100:.1f}%"
        )
    top_emotions_text = "\n".join(top_emotions) or "- 분석 결과 없음"

    presentation_text = f"""[PPT 발표용 분석 결과]

영상 제목: {result['title']}
영상 ID: {result['video_id']}
수집 댓글: {result['collected_comments']:,}개
한국어 분석 댓글: {result['analyzed_comments']:,}개

1. 최종 판정
- 조직적 활동 점수: {score['score']} / {score['max_score']}
- 등급: {score['grade']}
- 판정: {score['verdict']}

2. 항목별 점수
{item_scores_text}

3. 주요 분석값
- 댓글 재급증: {m['respike_days']}일
- TF-IDF 유사 댓글: {_pct(m['tfidf_near_share'])}
- Ko-SRoBERTa 의미 유사 댓글: {_pct(m['semantic_near_share'])}
- 가장 큰 AI 의미 군집: {_pct(m['largest_cluster_share'])}
- 반복 작성자 댓글: {_pct(m['heavy_author_share'])}

4. KOTE AI 감정 분석
- 긍정: {_pct(vd['긍정'])}
- 부정: {_pct(vd['부정'])}
- 중립: {_pct(vd['중립'])}

세부 감정 TOP 5
{top_emotions_text}

5. 이번 영상의 판정 근거
{reasons_text}

6. 발표에서 설명할 AI 활용
- Ko-SRoBERTa: 댓글 문장을 의미 벡터로 변환하여 의미 유사도 분석
- Ko-SRoBERTa + DBSCAN: 의미가 비슷한 댓글을 자동으로 군집화
- KcELECTRA + KOTE: 44개 세부 감정을 멀티라벨로 분석
- 최종 점수는 AI 분석값과 시간·반복성·작성자 행동 지표를 결합하여 산출

※ KOTE 감정 분석은 현재 120점 판정에는 직접 포함하지 않고 보조 분석으로 사용합니다.
"""

    # ---------------- ZIP 생성 ----------------
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("01_분석요약.csv", _csv_bytes(summary_df))
        z.writestr("02_점수산정.csv", _csv_bytes(score_df))
        z.writestr("03_KOTE_44감정.csv", _csv_bytes(emotion_df))
        z.writestr("04_긍정부정중립.csv", _csv_bytes(valence_df))
        z.writestr("05_일별댓글수.csv", _csv_bytes(daily_df))
        z.writestr("06_반복문구.csv", _csv_bytes(phrase_df))
        z.writestr("07_AI의미군집.csv", _csv_bytes(cluster_df))
        z.writestr("08_감정별대표댓글.csv", _csv_bytes(emotion_example_df))
        z.writestr("PPT_발표요약.txt", presentation_text.encode("utf-8-sig"))

    return buf.getvalue()




MAX_BATCH_VIDEOS = 10


def build_batch_result_zip(results, errors=None):
    """여러 영상의 비교표와 각 영상 상세 결과를 하나의 ZIP으로 묶는다."""
    errors = errors or []

    comparison_rows = []
    ai_rows = []
    emotion_rows = []

    for r in results:
        m = r["metrics"]
        s = r["score_result"]
        vd = r["valence_distribution"]
        top_emotion = (
            r["emotion_distribution"][0]["감정"]
            if r["emotion_distribution"] else "분석값 없음"
        )

        comparison_rows.append({
            "영상 ID": r["video_id"],
            "영상 제목": r["title"],
            "수집 댓글": r["collected_comments"],
            "분석 댓글": r["analyzed_comments"],
            "최종 점수": s["score"],
            "최대 점수": s["max_score"],
            "등급": s["grade"],
            "판정": s["verdict"],
        })

        ai_rows.append({
            "영상 ID": r["video_id"],
            "영상 제목": r["title"],
            "48시간 댓글(%)": round(m["share_48h"] * 100, 2),
            "댓글 재급증(일)": m["respike_days"],
            "TF-IDF 유사댓글(%)": round(m["tfidf_near_share"] * 100, 2),
            "Ko-SRoBERTa 의미유사(%)": (
                round(m["semantic_near_share"] * 100, 2)
                if not np.isnan(m["semantic_near_share"]) else np.nan
            ),
            "최대 AI 군집(%)": round(m["largest_cluster_share"] * 100, 2),
            "반복 작성자 댓글(%)": round(m["heavy_author_share"] * 100, 2),
        })

        emotion_rows.append({
            "영상 ID": r["video_id"],
            "영상 제목": r["title"],
            "긍정(%)": round(vd["긍정"] * 100, 2),
            "부정(%)": round(vd["부정"] * 100, 2),
            "중립(%)": round(vd["중립"] * 100, 2),
            "가장 많이 검출된 세부 감정": top_emotion,
        })

    comparison_df = pd.DataFrame(comparison_rows)
    ai_df = pd.DataFrame(ai_rows)
    emotion_df = pd.DataFrame(emotion_rows)
    error_df = pd.DataFrame(errors)

    ranked = sorted(
        results,
        key=lambda x: x["score_result"]["score"],
        reverse=True
    )
    comparison_text = "\n".join(
        f"- {i}. {r['title']} : {r['score_result']['score']}/120 "
        f"({r['score_result']['verdict']})"
        for i, r in enumerate(ranked, 1)
    )

    ppt_summary = f"""[전체 영상 비교 분석 - PPT용 요약]

분석 영상 수: {len(results)}개

1. 영상별 120점 결과
{comparison_text}

2. 비교할 핵심 AI 분석
- Ko-SRoBERTa 의미 유사 댓글 비율
- Ko-SRoBERTa 임베딩 + DBSCAN 최대 의미 군집 비율
- KcELECTRA + KOTE 긍정/부정/중립 및 44개 세부 감정

3. 비교할 비-AI 분석
- 댓글 재급증
- TF-IDF 유사댓글
- 반복 작성자 댓글
- 업로드 후 48시간 댓글 비율(보조 지표)

4. 발표 시 주의
- 120점은 본 프로젝트에서 정의한 조직적 댓글 패턴 탐색 기준
- 실제 조직적 조작 여부를 직접 증명하는 점수가 아니라,
  동일한 기준으로 여러 영상을 비교하기 위한 운영 점수
- KOTE 감정은 현재 120점에 직접 반영하지 않고 보조 분석으로 사용
"""

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("전체영상_점수비교.csv", _csv_bytes(comparison_df))
        z.writestr("전체영상_AI지표비교.csv", _csv_bytes(ai_df))
        z.writestr("전체영상_KOTE감정비교.csv", _csv_bytes(emotion_df))
        if not error_df.empty:
            z.writestr("분석오류목록.csv", _csv_bytes(error_df))
        z.writestr("PPT_전체비교요약.txt", ppt_summary.encode("utf-8-sig"))

        # 기존 단일 영상용 ZIP의 파일들을 영상별 폴더에 풀어서 포함
        for r in results:
            single_zip = build_ppt_result_zip(r)
            folder = f"{r['video_id']}_{re.sub(r'[^0-9A-Za-z가-힣_-]+', '_', r['title'])[:30]}"
            with zipfile.ZipFile(io.BytesIO(single_zip), "r") as single:
                for name in single.namelist():
                    z.writestr(f"{folder}/{name}", single.read(name))

    return buf.getvalue()


def comparison_dataframe(results):
    rows = []
    for r in results:
        m = r["metrics"]
        s = r["score_result"]
        vd = r["valence_distribution"]
        rows.append({
            "영상": r["title"],
            "영상 ID": r["video_id"],
            "댓글 수": r["collected_comments"],
            "최종 점수": s["score"],
            "등급": s["grade"],
            "재급증(일)": m["respike_days"],
            "TF-IDF(%)": round(m["tfidf_near_share"] * 100, 1),
            "AI 의미유사(%)": (
                round(m["semantic_near_share"] * 100, 1)
                if not np.isnan(m["semantic_near_share"]) else np.nan
            ),
            "최대 AI 군집(%)": round(m["largest_cluster_share"] * 100, 1),
            "반복작성자(%)": round(m["heavy_author_share"] * 100, 1),
            "긍정(%)": round(vd["긍정"] * 100, 1),
            "부정(%)": round(vd["부정"] * 100, 1),
            "중립(%)": round(vd["중립"] * 100, 1),
        })
    return pd.DataFrame(rows)


def render_video_detail(result):
    m = result["metrics"]
    score = result["score_result"]
    vd = result["valence_distribution"]

    st.markdown(f"### {result['title']}")
    st.caption(
        f"영상 ID: {result['video_id']} · "
        f"수집 {result['collected_comments']:,}개 · "
        f"분석 {result['analyzed_comments']:,}개"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("최종 점수", f"{score['score']} / 120")
    c2.metric("AI 의미 유사", _pct(m["semantic_near_share"]))
    c3.metric("최대 AI 군집", _pct(m["largest_cluster_share"]))
    c4.metric("반복 작성자", _pct(m["heavy_author_share"]))

    st.markdown(f"**판정:** {score['verdict']}")

    score_df = pd.DataFrame({
        "항목": list(score["item_scores"].keys()),
        "점수": list(score["item_scores"].values()),
    }).set_index("항목")
    st.bar_chart(score_df, y="점수", height=300)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown("**KOTE 긍정·부정·중립**")
        valence_df = pd.DataFrame({
            "분류": ["긍정", "부정", "중립"],
            "비율(%)": [
                vd["긍정"] * 100,
                vd["부정"] * 100,
                vd["중립"] * 100,
            ]
        }).set_index("분류")
        st.bar_chart(valence_df, y="비율(%)", height=280)

    with d2:
        st.markdown("**KOTE 세부 감정 TOP 10**")
        emo = pd.DataFrame(result["emotion_distribution"][:10])
        if not emo.empty:
            emo["비율(%)"] = emo["비율"] * 100
            st.dataframe(
                emo[["감정", "분류", "비율(%)"]].round({"비율(%)": 1}),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("KOTE 감정 분석값이 없습니다.")

    st.markdown("**날짜별 댓글 수**")
    daily = result["daily"].set_index("날짜")
    st.line_chart(daily, y="댓글 수", height=260)

    if result["top_clusters"]:
        st.markdown("**AI 의미 군집 대표 댓글**")
        for cluster in result["top_clusters"][:3]:
            with st.expander(
                f"군집 {cluster['cluster_id']} · {cluster['size']}개 댓글"
            ):
                for i, text in enumerate(cluster["examples"], 1):
                    st.write(f"{i}. {text}")

    single_zip = build_ppt_result_zip(result)
    safe_title = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", result["title"]).strip("_")
    st.download_button(
        "📦 이 영상 PPT용 결과 다운로드",
        data=single_zip,
        file_name=f"{result['video_id']}_{safe_title[:35]}_PPT분석결과.zip",
        mime="application/zip",
        key=f"single_{result['video_id']}",
        use_container_width=True,
    )


st.set_page_config(
    page_title="AI 유튜브 댓글 일괄 분석",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI 유튜브 댓글 조직적 패턴 일괄 분석")
st.caption(
    "YouTube 주소를 여러 개 입력하면 동일한 기준으로 자동 분석하고 "
    "영상별 120점, AI 의미 분석, KOTE 감정 결과를 한 번에 비교합니다."
)

api_key = ""
try:
    api_key = st.secrets.get("YOUTUBE_API_KEY", "")
except Exception:
    pass
api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")

if not api_key:
    with st.sidebar:
        st.subheader("YouTube API 설정")
        api_key = st.text_input("YouTube API Key", type="password")

st.info(
    "URL을 한 줄에 하나씩 입력하세요. 단일 영상도 가능하며, "
    f"한 번에 최대 {MAX_BATCH_VIDEOS}개까지 분석하도록 제한했습니다."
)

urls_text = st.text_area(
    "분석할 YouTube 영상 주소",
    height=220,
    placeholder=(
        "https://www.youtube.com/watch?v=AAAA\n"
        "https://www.youtube.com/watch?v=BBBB\n"
        "https://www.youtube.com/watch?v=CCCC"
    ),
)

start = st.button(
    "🚀 전체 영상 분석 시작",
    type="primary",
    use_container_width=True,
)

if start:
    urls = [line.strip() for line in urls_text.splitlines() if line.strip()]

    if not api_key:
        st.error("YouTube API Key가 필요합니다.")
    elif not urls:
        st.error("분석할 YouTube 주소를 한 개 이상 입력하세요.")
    elif len(urls) > MAX_BATCH_VIDEOS:
        st.error(
            f"한 번에 최대 {MAX_BATCH_VIDEOS}개까지 분석할 수 있습니다. "
            "URL 수를 줄여주세요."
        )
    else:
        progress = st.progress(0.0)
        status = st.empty()

        def update_progress(done, total, message):
            ratio = min(max(done / max(total, 1), 0.0), 1.0)
            progress.progress(ratio)
            status.write(message)

        try:
            batch = analyze_videos_batch(
                urls,
                api_key,
                progress_callback=update_progress
            )
            progress.progress(1.0)
            status.success("전체 분석이 완료되었습니다.")

            st.session_state["batch_results"] = batch["results"]
            st.session_state["batch_errors"] = batch["errors"]

        except Exception as e:
            st.error(f"전체 분석 중 오류가 발생했습니다: {e}")

results = st.session_state.get("batch_results", [])
errors = st.session_state.get("batch_errors", [])

if results:
    st.divider()
    st.header("1. 전체 영상 비교")

    comp = comparison_dataframe(results)
    st.dataframe(
        comp,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("영상별 최종 120점 비교")
    score_chart = comp[["영상", "최종 점수"]].copy().set_index("영상")
    st.bar_chart(score_chart, y="최종 점수", height=430)

    st.subheader("핵심 분석 지표 비교")
    ai_chart = comp[
        ["영상", "TF-IDF(%)", "AI 의미유사(%)", "최대 AI 군집(%)", "반복작성자(%)"]
    ].copy().set_index("영상")
    st.bar_chart(ai_chart, height=460)

    st.subheader("KOTE 긍정 · 부정 · 중립 비교")
    emotion_chart = comp[
        ["영상", "긍정(%)", "부정(%)", "중립(%)"]
    ].copy().set_index("영상")
    st.bar_chart(emotion_chart, height=420)

    st.divider()
    st.header("2. 영상별 상세 결과")

    for i, result in enumerate(results, 1):
        with st.expander(
            f"{i}. {result['title']} — {result['score_result']['score']}/120점",
            expanded=(i == 1),
        ):
            render_video_detail(result)

    st.divider()
    st.header("3. 전체 결과 다운로드")

    batch_zip = build_batch_result_zip(results, errors)
    st.download_button(
        "📦 전체 영상 PPT용 분석결과 ZIP 다운로드",
        data=batch_zip,
        file_name="youtube_batch_PPT_analysis_results.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    st.caption(
        "전체 점수 비교, AI 지표 비교, KOTE 감정 비교와 "
        "각 영상의 상세 CSV·대표 댓글·PPT 요약문이 한 ZIP에 들어갑니다."
    )

if errors:
    with st.expander(f"⚠️ 분석 중 오류/경고 {len(errors)}건 보기"):
        st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
