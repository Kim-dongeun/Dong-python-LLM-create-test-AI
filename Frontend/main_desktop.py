import streamlit as st
import ui_desktop as ui
import core_logic as cl

def main():
    # CSS for desktop optimization
    st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-right: 2rem;
        padding-left: 2rem;
        padding-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    cl.init_session_state()
    cl.init_cookies()  # 쿠키 초기화 추가
    ui.main()

if __name__ == "__main__":
    main()