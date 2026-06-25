#!/usr/bin/env python3
"""Generate a cryptographically secure API key for RAG_API_MASTER_KEY.

Usage:
    python scripts/generate_api_key.py
    python scripts/generate_api_key.py --length 48

Output: prints the key and its SHA-256 hash (the hash is what gets stored server-side).
"""
import argparse
import hashlib
import secrets
import string


def generate_key(length: int = 40) -> str:
    alphabet = string.ascii_letters + string.digits
    return "rag-" + "".join(secrets.choice(alphabet) for _ in range(length))

def main():
    parser = argparse.ArgumentParser(description="Generate RAG API key")
    parser.add_argument("--length", type=int, default=40, help="Key length (default: 40)")
    args = parser.parse_args()

    key = generate_key(args.length)
    key_hash = hashlib.sha256(key.encode()).hexdigest()

    print("\nGenerated API Key:")
    print(f"  RAG_API_MASTER_KEY={key}")
    print("\nSHA-256 Hash (for server-side storage):")
    print(f"  {key_hash}")
    print("\n⚠  Store the key in a secrets manager — NOT in source control.")
    print("   The hash is what the auth middleware compares against.\n")

if __name__ == "__main__":
    main()
