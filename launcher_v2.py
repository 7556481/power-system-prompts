import os
import sys
import subprocess


def main():
    print("🚀 Starting Prompt Evaluation Platform v2...")
    print("📍 URL: http://localhost:8502")
    print("⏹️ Press Ctrl+C to stop\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit",
            "run", "prompt_testing_v2.py",
            "--server.port", "8502"
        ])
    except KeyboardInterrupt:
        print("\n👋 Platform stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
