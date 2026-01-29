# launcher.py
import os
import sys
import subprocess


def main():
    print("🚀 Starting Prompt Evaluation Platform...")
    print("📍 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop\n")

    # 检查当前目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # 运行streamlit命令
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit",
            "run", "prompt_testing.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Platform stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()