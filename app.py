import os
import io
import zipfile
import re
import numpy as np
import pandas as pd
import streamlit as st

from analyzer import analyze_video


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


st.set_page_config(
    page_title="AI 유튜브 댓글 이상 패턴 분석",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 AI 유튜브 댓글 이상 패턴 분석")
st.caption(
    "YouTube 주소를 입력하면 댓글의 시간 집중, 반복성, AI 의미 유사도, "
    "작성자 반복과 KOTE 기반 44개 세부 감정을 분석합니다."
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
        api_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="입력한 키는 화면에 표시되지 않습니다."
        )

url = st.text_input(
    "분석할 YouTube 영상 주소",
    placeholder="https://www.youtube.com/watch?v=..."
)

analyze = st.button(
    "댓글 분석하기",
    type="primary",
    use_container_width=True
)

if analyze:
    if not api_key:
        st.error("YouTube API Key를 먼저 입력하세요.")
        st.stop()

    if not url.strip():
        st.error("YouTube 영상 주소를 입력하세요.")
        st.stop()

    try:
        with st.spinner(
            "댓글을 수집하고 AI가 분석하고 있습니다. "
            "처음 실행할 때는 AI 모델 다운로드 때문에 시간이 더 걸릴 수 있습니다."
        ):
            result = analyze_video(url, api_key)

        st.success("분석이 완료되었습니다.")

        st.subheader(result["title"])
        st.caption(
            f"영상 ID: {result['video_id']} · "
            f"수집 댓글 {result['collected_comments']:,}개 · "
            f"한국어 분석 댓글 {result['analyzed_comments']:,}개"
        )

        # ---------------- 기존 이상 패턴 지표 ----------------
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

        # ---------------- 120점 판정 ----------------
        score = result["score_result"]

        st.subheader("조직적 댓글 활동 판정")
        s1, s2 = st.columns([1, 2])
        s1.metric("조직적 활동 점수", f"{score['score']} / {score['max_score']}")
        s2.markdown(f"### {score['verdict']}")
        s2.caption(f"판정 등급: {score['grade']}")

        st.progress(score["score"] / score["max_score"])

        score_df = pd.DataFrame(
            {
                "항목": list(score["item_scores"].keys()),
                "점수": list(score["item_scores"].values()),
            }
        ).set_index("항목")
        st.bar_chart(score_df, y="점수", height=360)

        with st.expander("항목별 판정 기준과 근거 보기", expanded=True):
            criteria = pd.DataFrame([
                ["댓글 재급증", "0일=0점 / 1일=10점 / 2일 이상=20점"],
                ["AI 의미 유사 댓글", "<10%=0점 / 10~20%=10점 / ≥20%=20점"],
                ["최대 AI 군집", "<3%=0점 / 3~7%=10점 / ≥7%=20점"],
                ["반복 작성자 댓글", "<5%=0점 / 5~10%=10점 / ≥10%=20점"],
                ["TF-IDF 유사 댓글", "<3%=0점 / 3~8%=10점 / ≥8%=20점"],
                ["복합 신호", "시간·내용·작성자 이상이 동시에 발생하면 20점"],
            ], columns=["판정 항목", "점수 기준"])
            st.dataframe(criteria, use_container_width=True, hide_index=True)

            if score["reasons"]:
                st.markdown("**이번 영상의 주요 판정 근거**")
                for r in score["reasons"]:
                    st.write(f"• {r}")

        st.caption(
            "점수 구간: 0~29 정상 / 30~49 주의 / 50~69 조직적 활동 의심 / "
            "70~120 조직적 활동 강하게 의심. "
            "본 점수는 이 과제에서 정의한 댓글 패턴 판정 기준입니다."
        )

        chart = pd.DataFrame({
            "지표": [
                "48시간 댓글",
                "TF-IDF 유사 댓글",
                "AI 의미 유사 댓글",
                "최대 AI 군집",
                "반복 작성자 댓글",
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
        daily = result["daily"].set_index("날짜")
        st.line_chart(daily, y="댓글 수", height=330)

        # ---------------- KOTE 세부 감정 ----------------
        st.divider()
        st.header("2. KOTE 기반 44개 세부 감정 분석")

        st.caption(
            f"KcELECTRA + KOTE 44감정 멀티라벨 모델 사용 · "
            f"표본 {result['emotion_sample_size']:,}개 댓글 · "
            f"감정 채택 임계값 {result['emotion_threshold']:.2f}"
        )

        st.info(
            "한 댓글에 '화남/분노 + 불평/불만 + 의심/불신'처럼 여러 감정이 동시에 "
            "검출될 수 있습니다. 따라서 아래 44개 세부 감정 비율의 합은 100%가 아닐 수 있습니다."
        )

        emotion_df = pd.DataFrame(result["emotion_distribution"])
        if not emotion_df.empty:
            display_emotion = emotion_df.copy()
            display_emotion["비율(%)"] = display_emotion["비율"] * 100

            top15 = display_emotion.head(15).set_index("감정")
            st.subheader("많이 나타난 세부 감정 TOP 15")
            st.bar_chart(top15, y="비율(%)", height=520)

            with st.expander("44개 감정 전체 결과 보기"):
                table_view = display_emotion[
                    ["감정", "분류", "댓글 수", "비율(%)", "평균 확률"]
                ].copy()
                table_view["비율(%)"] = table_view["비율(%)"].round(1)
                table_view["평균 확률"] = table_view["평균 확률"].round(3)
                st.dataframe(
                    table_view,
                    use_container_width=True,
                    hide_index=True
                )

        # ---------------- 긍정/부정/중립 ----------------
        st.subheader("긍정 · 부정 · 중립 요약")

        vd = result["valence_distribution"]
        v1, v2, v3 = st.columns(3)
        v1.metric("긍정", f"{vd['긍정']:.1%}")
        v2.metric("부정", f"{vd['부정']:.1%}")
        v3.metric("중립", f"{vd['중립']:.1%}")

        valence_df = pd.DataFrame({
            "분류": ["긍정", "부정", "중립"],
            "비율(%)": [
                vd["긍정"] * 100,
                vd["부정"] * 100,
                vd["중립"] * 100,
            ]
        }).set_index("분류")
        st.bar_chart(valence_df, y="비율(%)", height=330)

        st.caption(
            "긍정·부정·중립은 KOTE의 세부 감정을 valence 그룹으로 묶은 요약입니다. "
            "각 댓글에서 긍정·부정·중립 그룹별 가장 높은 감정 확률을 비교해 하나로 분류합니다."
        )

        # ---------------- 대표 감정 댓글 ----------------
        st.subheader("세부 감정별 대표 댓글")
        examples = result["top_emotion_examples"]

        if not examples:
            st.write("표시할 대표 감정 댓글이 없습니다.")
        else:
            for emotion, rows in examples.items():
                with st.expander(f"{emotion} 대표 댓글"):
                    for i, row in enumerate(rows, 1):
                        st.write(
                            f"{i}. **{row['score']:.1%}** — {row['text']}"
                        )

        # ---------------- 기존 문구/군집 결과 ----------------
        st.divider()
        st.header("3. 반복 문구와 AI 의미 군집")

        if result["top_phrases"]:
            st.subheader("많이 반복된 동일 문구")
            phrase_df = pd.DataFrame(result["top_phrases"])
            phrase_df.columns = ["댓글", "반복 횟수"]
            st.dataframe(
                phrase_df,
                use_container_width=True,
                hide_index=True
            )

        st.subheader("AI가 발견한 의미가 비슷한 댓글 집단")
        if not result["top_clusters"]:
            st.write("현재 기준에서 뚜렷한 의미 군집이 발견되지 않았습니다.")
        else:
            for cluster in result["top_clusters"]:
                with st.expander(
                    f"군집 {cluster['cluster_id']} · {cluster['size']}개 댓글"
                ):
                    for i, text in enumerate(cluster["examples"], 1):
                        st.write(f"{i}. {text}")


        # ---------------- 발표자료용 결과 다운로드 ----------------
        st.divider()
        st.header("4. 발표자료용 분석 결과 다운로드")
        st.write(
            "이 영상의 점수, AI 분석값, 44개 감정, 일별 댓글 수, "
            "AI 의미 군집과 대표 댓글을 한 번에 저장할 수 있습니다."
        )

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

        st.caption(
            "ZIP 안에는 분석요약, 120점 점수산정, KOTE 44개 감정, "
            "긍정·부정·중립, 일별 댓글 수, 반복 문구, AI 의미 군집, "
            "감정별 대표 댓글, PPT 발표용 요약문이 포함됩니다."
        )

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.caption(
            "댓글 비활성화, 비공개 영상, API 할당량, 잘못된 API 키, "
            "또는 배포 서버의 메모리 부족 등이 원인일 수 있습니다."
        )
