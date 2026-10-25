"""
Admin Setup Utility: Generate a secure password hash and .env file for AI Image Detector.
Run this script to configure or reset your single admin account credentials.
"""

import getpass
import os
import secrets
from werkzeug.security import generate_password_hash


def main():
    print("=" * 65)
    print(" 🔐 AI Image Detector - Admin Account Configuration Helper")
    print("=" * 65)
    print("This utility generates a cryptographically hashed password and a")
    print("secure random SECRET_KEY to store in your local .env file.\n")

    # Prompt for username
    default_user = "admin"
    username_input = input(f"Enter admin username [default: {default_user}]: ").strip()
    admin_username = username_input if username_input else default_user

    # Prompt for password securely without echoing
    while True:
        password = getpass.getpass("Enter new admin password: ")
        if not password:
            print("Password cannot be empty. Please try again.")
            continue
        confirm = getpass.getpass("Confirm admin password: ")
        if password != confirm:
            print("Passwords do not match. Please try again.\n")
            continue
        break

    # Generate hash using modern scrypt
    hashed_pw = generate_password_hash(password, method="scrypt")
    secret_key = secrets.token_hex(32)

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env_content = (
        f"# AI Image Detector - Secret Configuration\n"
        f"# Generated automatically on admin setup\n"
        f"SECRET_KEY={secret_key}\n"
        f"ADMIN_USERNAME={admin_username}\n"
        f"ADMIN_PASSWORD_HASH={hashed_pw}\n"
        f"MAX_CONTENT_LENGTH=10485760\n"
    )

    print("\n" + "-" * 65)
    print("✅ Password hash generated successfully!")
    print(f"Username : {admin_username}")
    print(f"Hash     : {hashed_pw[:35]}...")
    print("-" * 65)

    save_choice = input(f"Write these credentials directly to '{env_path}'? [Y/n]: ").strip().lower()
    if save_choice in ("", "y", "yes"):
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(env_content)
        print(f"\n🎉 Saved credentials to '{env_path}'!")
        print("Note: The '.env' file is already in .gitignore and will never be committed to git.")
    else:
        print("\nHere are your environment variables to add manually to your .env file:\n")
        print(env_content)


if __name__ == "__main__":
    main()
