import streamlit as st
import time
from browser_detection import browser_detection_engine
import main_desktop
import main_mobile
import core_logic as cl

# 기본 페이지 설정
st.set_page_config(
    page_title="AI 학습 플랫폼",  # 기본 제목
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto"  # 기본 사이드바 상태
)

# 브라우저 정보 가져오기
browser_info = browser_detection_engine(singleRun=True)
time.sleep(0.5)
is_mobile = browser_info.get('isMobile', False)
is_tablet = browser_info.get('isTablet', False)
is_desktop = browser_info.get('isDesktop', False)

# 현재 URL의 query params 가져오기
query_params = st.query_params
mode = query_params.get("mode", ["desktop"])[0]  # 기본값은 'desktop'

# 페이지 타이틀과 사이드바 상태 설정
if mode == "mobile":
    page_title = "AI 학습 플랫폼 (모바일)"
    sidebar_state = "collapsed"
else:
    page_title = "AI 학습 플랫폼"
    sidebar_state = "expanded"

# JavaScript를 사용하여 URL 파라미터를 설정하고 페이지를 다시 로드하는 함수
def update_url_params(mode):
    st.write(
        f"""
        <script>
            function updateURL() {{
                const params = new URLSearchParams(window.location.search);
                params.set('mode', '{mode}');
                window.location.search = params.toString();
            }}
            updateURL();
        </script>
        """,
        unsafe_allow_html=True
    )

def run():
    if is_mobile or is_tablet:
        main_mobile.main()
    if is_desktop:
        main_desktop.main()

if __name__ == "__main__":
    cl.init_session_state()
    cl.init_cookies()  # 쿠키 초기화 추가
    run()
