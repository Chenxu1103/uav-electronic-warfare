import os
import sys

def fix_imports():
    # Get the current working directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Add the current directory to Python path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"Added {current_dir} to Python path")
    else:
        print(f"{current_dir} already in Python path")
    
    # Print the current Python path for debugging
    print("Current Python path:")
    for path in sys.path:
        print(f"  {path}")

if __name__ == "__main__":
    fix_imports() 