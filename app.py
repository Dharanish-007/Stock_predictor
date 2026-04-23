import os
import sys

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main function from the streamlit app
from app.streamlit_app import main

if __name__ == "__main__":
    main()
