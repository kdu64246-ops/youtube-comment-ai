import os
import io
import zipfile
import re
import numpy as np
import pandas as pd
import streamlit as st

from analyzer import analyze_video, load_single_checkpoint

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




CHECKPOINT_PATH = os.path.join("data", "latest_single_result.pkl")


def render_result(result):
    st.success("분석 결과를 불러왔습니다.")

    st.subheader(result["title"])
    st.caption(
        f"영상 ID: {result['video_id']} · "
        f"수집 댓글 {result['collected_comments']:,}개 · "
        f"한국어 분석 댓글 {result['analyzed_comments']:,}개"
    )

    m = result["metrics"]

    st.header("1. 댓글 이상 패턴 분석")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("48시간 내 댓글", f"{m['share_48h']:.1%}")
    c2.metric("댓글 재급증", f"{m['respike_days']}일")
    c3.metric("AI 의미 유사 댓글", f"{m['semantic_near_share']:.1%}")
    c4.metric("AI 의미 군집", f"{m['semantic_cluster_count']}개")

    c5, c6, c7 = st.columns(3)
    c5.metric("TF-IDF 유사 댓글", f"{m['tfidf_near_share']:.1%}")
    c6.metric("최대 AI 군집", f"{m['largest_cluster_share']:.1%}")
    c7.metric("반복 작성자 댓글", f"{m['heavy_author_share']:.1%}")

    score = result["score_result"]
    st.subheader("조직적 댓글 활동 판정")
    s1, s2 = st.columns([1, 2])
    s1.metric("조직적 활동 점수", f"{score['score']} / {score['max_score']}")
    s2.markdown(f"### {score['verdict']}")
    s2.caption(f"판정 등급: {score['grade']}")
    st.progress(score["score"] / score["max_score"])

    score_df = pd.DataFrame({
        "항목": list(score["item_scores"].keys()),
        "점수": list(score["item_scores"].values()),
    }).set_index("항목")
    st.bar_chart(score_df, y="점수", height=360)

    with st.expander("항목별 판정 근거 보기", expanded=True):
        if score["reasons"]:
            for reason in score["reasons"]:
                st.write(f"• {reason}")
        else:
            st.write("점수 기준을 넘은 주요 이상 신호가 없습니다.")

    chart = pd.DataFrame({
        "지표": [
            "48시간 댓글", "TF-IDF 유사 댓글", "AI 의미 유사 댓글",
            "최대 AI 군집", "반복 작성자 댓글"
        ],
        "비율(%)": [
            m["share_48h"] * 100,
            m["tfidf_near_share"] * 100,
            m["semantic_near_share"] * 100,
            m["largest_cluster_share"] * 100,
            m["heavy_author_share"] * 100,
        ]
    }).set_index("지표")
    st.bar_chart(chart, y="비율(%)", height=420)

    st.subheader("날짜별 댓글 수")
    st.line_chart(result["daily"].set_index("날짜"), y="댓글 수", height=330)

    st.divider()
    st.header("2. KOTE 기반 44개 세부 감정 분석")

    if result.get("emotion_error"):
        st.warning(
            "핵심 120점 분석은 정상 완료됐지만 KOTE 감정 분석에서 오류가 발생했습니다. "
            "점수 결과와 다운로드는 그대로 사용할 수 있습니다."
        )
        st.code(result["emotion_error"])
    elif result.get("emotion_distribution"):
        st.caption(
            f"KcELECTRA + KOTE · 표본 {result['emotion_sample_size']:,}개 · "
            f"임계값 {result['emotion_threshold']:.2f}"
        )

        emotion_df = pd.DataFrame(result["emotion_distribution"])
        emotion_df["비율(%)"] = emotion_df["비율"] * 100

        top15 = emotion_df.head(15).set_index("감정")
        st.subheader("많이 나타난 세부 감정 TOP 15")
        st.bar_chart(top15, y="비율(%)", height=520)

        vd = result["valence_distribution"]
        v1, v2, v3 = st.columns(3)
        v1.metric("긍정", f"{vd['긍정']:.1%}")
        v2.metric("부정", f"{vd['부정']:.1%}")
        v3.metric("중립", f"{vd['중립']:.1%}")

        with st.expander("44개 감정 전체 결과 보기"):
            table_view = emotion_df[
                ["감정", "분류", "댓글 수", "비율(%)", "평균 확률"]
            ].copy()
            table_view["비율(%)"] = table_view["비율(%)"].round(1)
            table_view["평균 확률"] = table_view["평균 확률"].round(3)
            st.dataframe(table_view, use_container_width=True, hide_index=True)
    else:
        st.info("KOTE 감정 분석 결과가 없습니다.")

    st.divider()
    st.header("3. 반복 문구와 AI 의미 군집")

    if result["top_phrases"]:
        phrase_df = pd.DataFrame(result["top_phrases"])
        phrase_df.columns = ["댓글", "반복 횟수"]
        st.dataframe(phrase_df, use_container_width=True, hide_index=True)

    if result["top_clusters"]:
        for cluster in result["top_clusters"]:
            with st.expander(
                f"군집 {cluster['cluster_id']} · {cluster['size']}개 댓글"
            ):
                for i, text in enumerate(cluster["examples"], 1):
                    st.write(f"{i}. {text}")
    else:
        st.write("뚜렷한 AI 의미 군집이 없습니다.")

    st.divider()
    st.header("4. 발표자료용 분석 결과 다운로드")

    export_zip = build_ppt_result_zip(result)
    safe_title = re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", result["title"]).strip("_")
    safe_title = safe_title[:40] if safe_title else result["video_id"]

    st.download_button(
        label="📦 PPT용 분석결과 ZIP 다운로드",
        data=export_zip,
        file_name=f"{result['video_id']}_{safe_title}_PPT분석결과.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )


st.set_page_config(
    page_title="AI 유튜브 댓글 이상 패턴 분석",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI 유튜브 댓글 이상 패턴 분석")
st.caption(
    "단일 YouTube 영상을 분석합니다. 핵심 120점 결과를 먼저 저장하므로 "
    "KOTE 감정 분석에 문제가 생겨도 점수 결과는 유지됩니다."
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

url = st.text_input(
    "분석할 YouTube 영상 주소",
    placeholder="https://www.youtube.com/watch?v=..."
)

# 세션이 끊긴 뒤에도 서버에 저장된 최근 결과를 복구
saved_result = None
try:
    saved_result = load_single_checkpoint(CHECKPOINT_PATH)
except Exception:
    saved_result = None

if "single_result" not in st.session_state and saved_result is not None:
    if st.button("🔄 최근 저장된 분석 결과 불러오기", use_container_width=True):
        st.session_state["single_result"] = saved_result
        st.rerun()

analyze = st.button(
    "댓글 분석하기",
    type="primary",
    use_container_width=True
)

if analyze:
    if not api_key:
        st.error("YouTube API Key를 먼저 입력하세요.")
    elif not url.strip():
        st.error("YouTube 영상 주소를 입력하세요.")
    else:
        progress = st.progress(0.0)
        status = st.empty()

        def progress_callback(value, message):
            progress.progress(min(max(float(value), 0.0), 1.0))
            status.write(message)

        try:
            result = analyze_video(
                url,
                api_key,
                checkpoint_path=CHECKPOINT_PATH,
                progress_callback=progress_callback,
            )
            st.session_state["single_result"] = result
            progress.progress(1.0)
            status.success("분석이 완료되었습니다.")
            st.rerun()
        except Exception as e:
            st.error(f"분석 중 오류가 발생했습니다: {e}")

            # 분석 도중 마지막 단계에서 실패해도 핵심 체크포인트가 있으면 복구
            try:
                recovered = load_single_checkpoint(CHECKPOINT_PATH)
            except Exception:
                recovered = None

            if recovered is not None:
                st.warning(
                    "전체 분석 과정에서 오류가 있었지만, 그 전에 저장된 핵심 결과가 있습니다."
                )
                st.session_state["single_result"] = recovered

result = st.session_state.get("single_result")
if result is not None:
    render_result(result)
