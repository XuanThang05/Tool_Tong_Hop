import sys
import os

# Add the project root to sys.path so 'src' can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



from src.ui_app import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
