

[KOTE 44감정 업데이트]
- 기존 긍정/부정 이진 감성분석을 제거했습니다.
- searle-j/kote_for_easygoing_people (KcELECTRA + KOTE) 멀티라벨 모델을 사용합니다.
- 44개 세부 감정을 분석합니다.
- 한 댓글에서 여러 감정을 동시에 검출할 수 있습니다.
- 세부 감정은 threshold 0.40 이상을 검출 감정으로 집계합니다.
- 세부 감정을 KOTE valence 분류에 따라 긍정/부정/중립으로 다시 요약합니다.
- 웹 배포 메모리를 고려해 감정 분석은 최대 1,500개 한국어 댓글 표본을 사용합니다.

GitHub에 이미 배포한 경우:
1) analyzer.py 교체
2) app.py 교체
3) requirements.txt 확인
4) Commit changes
5) Streamlit Cloud가 자동으로 재배포할 때까지 로그 확인
