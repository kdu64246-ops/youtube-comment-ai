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
    "작성자 반복, 감정 경향을 분석합니다."
)

st.info(
    "이 서비스는 댓글부대의 존재를 확정하는 도구가 아닙니다. "
    "여러 댓글 패턴 중 사람이 추가로 확인할 후보를 선별하는 시뮬레이션입니다."
)

# Streamlit Cloud Secrets 또는 환경 변수에서 API 키를 먼저 찾음
api_key = ""

try:
    api_key = st.secrets.get("YOUTUBE_API_KEY", "")
except Exception:
    pass

api_key = api_key or os.environ.get("YOUTUBE_API_KEY", "")

# 로컬 테스트에서 Secret이 없을 경우 직접 입력 가능
if not api_key:
    with st.sidebar:
        st.subheader("YouTube API 설정")
        api_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="입력한 키는 이 화면에 표시되지 않습니다."
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
            "처음 실행할 때는 AI 모델을 불러와 시간이 더 걸릴 수 있습니다."
        ):
            result = analyze_video(url, api_key)

        st.success("분석이 완료되었습니다.")

        st.subheader(result["title"])
        st.caption(
            f"영상 ID: {result['video_id']} · "
            f"수집 댓글 {result['collected_comments']:,}개 · "
            f"한국어 분석 댓글 {result['analyzed_comments']:,}개"
        )

        # ---------------- 핵심 지표 ----------------
        m = result["metrics"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("48시간 내 댓글", f"{m['share_48h']:.1%}")
        c2.metric("댓글 재급증", f"{m['respike_days']}일")
        c3.metric("AI 의미 유사 댓글", f"{m['semantic_near_share']:.1%}")
        c4.metric("AI 의미 군집", f"{m['semantic_cluster_count']}개")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("TF-IDF 유사 댓글", f"{m['tfidf_near_share']:.1%}")
        c6.metric("최대 AI 군집", f"{m['largest_cluster_share']:.1%}")
        c7.metric("반복 작성자 댓글", f"{m['heavy_author_share']:.1%}")

        if np.isnan(m["positive"]):
            c8.metric("긍정 댓글", "N/A")
        else:
            c8.metric("긍정 댓글", f"{m['positive']:.1%}")

        # ---------------- 시뮬레이션 결과 ----------------
        st.divider()
        st.subheader("시뮬레이션 분석 결과")

        if len(result["signals"]) >= 3:
            st.warning(result["level"])
        elif len(result["signals"]) >= 1:
            st.info(result["level"])
        else:
            st.success(result["level"])

        if result["signals"]:
            for s in result["signals"]:
                st.write(f"• {s}")
        else:
            st.write("현재 탐색 기준에서 두드러진 이상 신호가 발견되지 않았습니다.")

        st.caption(
            "위 결과는 프로젝트 시뮬레이션용 탐색 기준입니다. "
            "실제 조직적 조작 여부를 증명하거나 특정 작성자를 식별하는 결과가 아닙니다."
        )

        # ---------------- 세로형 비교 그래프 ----------------
        st.divider()
        st.subheader("주요 지표")

        chart = pd.DataFrame({
            "지표": [
                "48시간 댓글",
                "TF-IDF 유사 댓글",
                "AI 의미 유사 댓글",
                "최대 AI 군집",
                "반복 작성자 댓글",
                "긍정 댓글",
            ],
            "비율(%)": [
                m["share_48h"] * 100,
                m["tfidf_near_share"] * 100,
                m["semantic_near_share"] * 100,
                m["largest_cluster_share"] * 100,
                m["heavy_author_share"] * 100,
                0 if np.isnan(m["positive"]) else m["positive"] * 100,
            ]
        }).set_index("지표")

        st.bar_chart(chart, y="비율(%)", height=430)

        # ---------------- 시간별 댓글 ----------------
        st.subheader("날짜별 댓글 수")
        daily = result["daily"].set_index("날짜")
        st.line_chart(daily, y="댓글 수", height=350)

        # ---------------- 반복 문구 ----------------
        if result["top_phrases"]:
            st.subheader("많이 반복된 동일 문구")
            phrase_df = pd.DataFrame(result["top_phrases"])
            phrase_df.columns = ["댓글", "반복 횟수"]
            st.dataframe(
                phrase_df,
                use_container_width=True,
                hide_index=True
            )

        # ---------------- AI 군집 ----------------
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
            "댓글이 비활성화된 영상, 비공개 영상, API 할당량 소진, "
            "잘못된 API 키 등이 원인일 수 있습니다."
        )
