#!/usr/bin/env python3
"""
Setup Verification Script
Checks if the course environment is properly configured.
"""

import sys


def print_header(text: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def check_python_version() -> bool:
    """Check Python version."""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    success = version >= (3, 11)

    symbol = "✓" if success else "✗"
    print(f"{symbol} Python version: {version_str}")

    if not success:
        print("  ⚠️  Python 3.11+ required")

    return success


def check_package(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        module = __import__(package_name)
        version = getattr(module, '__version__', 'installed')
        print(f"✓ {package_name:30s} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:30s} NOT INSTALLED")
        return False


def check_gpu() -> bool:
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            print(f"✓ GPU Support: {gpu_count}x {gpu_name}")
            return True
        else:
            print("⚠️  GPU: Not available (CPU only)")
            return False
    except:
        print("✗ PyTorch GPU check failed")
        return False


def check_env_vars() -> None:
    """Check environment variables."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_keys = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
        "HF_TOKEN": os.getenv("HF_TOKEN"),
    }

    for key, value in api_keys.items():
        if value:
            print(f"✓ {key:30s} configured")
        else:
            print(f"⚠️  {key:30s} not configured")


def main():
    """Run all verification checks."""
    print_header("LLM Course Environment Verification")

    # Python version
    all_good = check_python_version()

    # Core packages
    print_header("Core Packages")

    core_packages = [
        "torch",
        "transformers",
        "datasets",
        "langchain",
        "peft",
        "sentence_transformers",
        "openai",
        "anthropic",
    ]

    for package in core_packages:
        result = check_package(package)
        all_good = all_good and result

    # GPU Check
    print_header("Hardware")
    check_gpu()

    # Vector Databases
    print_header("Vector Databases")

    db_packages = ["chromadb", "faiss", "weaviate"]
    for package in db_packages:
        check_package(package)

    # Agents
    print_header("Agent Frameworks")

    agent_packages = ["langgraph", "crewai"]
    for package in agent_packages:
        check_package(package)

    # Jupyter
    print_header("Development Tools")
    check_package("jupyter")
    check_package("jupyterlab")

    # Environment Variables
    print_header("API Keys")
    check_env_vars()

    # Summary
    print_header("Summary")

    if all_good:
        print("✅ All core packages installed successfully!")
        print("You're ready to start the course! 🚀")
    else:
        print("⚠️  Some packages are missing.")
        print("Run: pip install -r requirements.txt")

    print("\n💡 Tip: If you see missing API keys, copy .env.example to .env")
    print("        and add your API keys there.")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Error during verification: {e}")
        print("Please check your installation.")
        sys.exit(1)
