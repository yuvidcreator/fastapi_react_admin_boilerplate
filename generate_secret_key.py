import secrets
import argparse

"""
Custom length (32 bytes = 256 bits):
python generate_secret_key.py --length 32

Output for .env file:
python generate_secret_key.py --env
"""

def generate_secret_key(length=64):
    """
    Generate a cryptographically secure random secret key.
    
    Args:
        length (int): Length of the key in bytes. Default is 64 (512 bits)
    
    Returns:
        str: Hexadecimal representation of the secret key
    """
    if length < 16:
        print("Warning: Very short keys are not secure. Consider using at least 32 bytes.")
    
    # Generate cryptographically secure random bytes
    secret_bytes = secrets.token_bytes(length)
    
    # Convert to hexadecimal string
    secret_key_hex = secret_bytes.hex()
    
    return secret_key_hex

def main():
    parser = argparse.ArgumentParser(description='Generate a cryptographically secure secret key')
    parser.add_argument('--length', type=int, default=64, 
                       help='Length of the key in bytes (default: 64)')
    parser.add_argument('--env', action='store_true',
                       help='Output in .env file format (KEY=value)')
    
    args = parser.parse_args()
    
    # Generate the secret key
    secret_key = generate_secret_key(args.length)
    
    # Output the result
    if args.env:
        print(f"SECRET_KEY={secret_key}")
    else:
        print(f"Generated secret key ({args.length} bytes):")
        print(secret_key)
        
        # Show the length info
        print(f"\nKey info:")
        print(f"- Bytes: {args.length}")
        print(f"- Bits: {args.length * 8}")
        print(f"- Hex characters: {len(secret_key)}")

if __name__ == "__main__":
    main()