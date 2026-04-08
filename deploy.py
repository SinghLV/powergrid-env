import sys
import getpass
from huggingface_hub import HfApi

def main():
    print("=" * 40)
    print("🚀 PowerGridEnv Hugging Face Deployer")
    print("=" * 40)
    print()

    token = getpass.getpass("Please enter your Hugging Face Token (starting with hf_): ")
    if not token.startswith("hf_"):
        print("\n❌ Invalid token. Hugging Face tokens usually start with 'hf_'.")
        sys.exit(1)

    api = HfApi(token=token)
    username = "lakshyavijay"
    repo_id = f"{username}/powergrid-env"

    print(f"\n[1/3] Creating Space '{repo_id}'...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
        print("✅ Space created (or already exists).")
    except Exception as e:
        print(f"\n❌ Error creating space: {e}")
        print("Please make sure your token has the 'Write' permission!")
        sys.exit(1)

    print("\n[2/3] Uploading project files...")
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type="space",
            ignore_patterns=[".git", ".venv", ".env", "__pycache__", "*.pyc", ".gitignore", "deploy.py"]
        )
        print("✅ Files uploaded successfully.")
    except Exception as e:
        print(f"\n❌ Error uploading files: {e}")
        sys.exit(1)

    print("\n[3/3] Setting OPENAI_API_KEY Secret...")
    print("⚠️  You will need to manually add your OPENAI_API_KEY or HF_TOKEN in the Space Settings later.")

    print("\n" + "=" * 40)
    print("🎉 DEPLOYMENT COMPLETE!")
    print(f"👉 Your environment is live at: https://huggingface.co/spaces/{repo_id}")
    print("=" * 40)

if __name__ == "__main__":
    main()
