import streamlit as st
import ui_mobile as ui
import core_logic as cl

def main():
    # CSS for mobile optimization
    st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 95%;
        padding-top: 2rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    cl.init_session_state()
    cl.init_cookies()  # 쿠키 초기화 추가
    ui.main()

if __name__ == "__main__":
    main()