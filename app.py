import os
import numpy as np
import pandas as pd
import streamlit as st

from analyzer import analyze_video

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

        st.subheader("이상 신호")
        st.write(f"**{result['level']}**")
        if result["signals"]:
            for s in result["signals"]:
                st.write(f"• {s}")
        else:
            st.write("현재 탐색 기준에서 두드러진 이상 신호가 발견되지 않았습니다.")

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

    except Exception as e:
        st.error(f"분석 중 오류가 발생했습니다: {e}")
        st.caption(
            "댓글 비활성화, 비공개 영상, API 할당량, 잘못된 API 키, "
            "또는 배포 서버의 메모리 부족 등이 원인일 수 있습니다."
        )
