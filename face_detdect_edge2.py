"""
Multi-Mode Entry Point:
- When launched by Streamlit Cloud (or 'streamlit run face_detdect_edge2.py'):
  Automatically runs the interactive Web Application.
- When launched from terminal ('python face_detdect_edge2.py'):
  Automatically runs the high-performance local OpenCV engine for PC/Raspberry Pi.
"""
import sys

def is_streamlit_running():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return "streamlit" in sys.argv[0] or "streamlit" in sys.modules

if is_streamlit_running():
    # Running inside Streamlit (Streamlit Cloud or local Streamlit runner)
    with open("app.py", "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
else:
    # Running via standard Python CLI ('python face_detdect_edge2.py')
    from main import main
    main()