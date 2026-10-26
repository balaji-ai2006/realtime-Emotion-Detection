"""
Legacy entry point for face_detdect_edge2.py
Automatically routes to the new, highly optimized main.py engine
compatible with PC and Raspberry Pi.
"""
from main import main

if __name__ == "__main__":
    main()