import subprocess
import sys
import os


def main():
    """
    Main entry point for the Clinical Trials Agent functionality.
    Launches the Gradio UI.
    """
    print("🚀 Launching Clinical Trials Recruitment Agent...")

    # Get the directory handling for robust path resolution
    base_dir = os.path.dirname(os.path.abspath(__file__))

    try:
        # Run the Gradio UI module
        # We use sys.executable to ensure we use the same python environment
        cmd = [sys.executable, "-m", "gradio_ui.gradio_main"]
        subprocess.run(cmd, check=True, cwd=base_dir)
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
