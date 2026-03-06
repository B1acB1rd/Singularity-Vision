"""
Encryption Utilities - Secure storage for sensitive data
"""
import os
import base64
import hashlib
import secrets
from typing import Optional
from pathlib import Path

# Try to use cryptography library, fall back to simple obfuscation
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class EncryptionManager:
    """
    Manages encryption/decryption of sensitive data.
    
    Uses Fernet symmetric encryption when available,
    with a machine-specific key derived from hardware identifiers.
    """
    
    def __init__(self, key_file: Optional[str] = None):
        self.key_file = key_file or os.path.join(
            os.path.expanduser("~"),
            ".singularity-vision",
            ".keyring"
        )
        self._key: Optional[bytes] = None
    
    def _get_machine_id(self) -> bytes:
        """Get a machine-specific identifier for key derivation"""
        # Combine multiple sources for uniqueness
        components = []
        
        # Username
        components.append(os.environ.get("USERNAME", os.environ.get("USER", "unknown")))
        
        # Home directory
        components.append(os.path.expanduser("~"))
        
        # Try to get machine identifier
        try:
            import platform
            components.append(platform.node())
            components.append(platform.machine())
        except:
            pass
        
        # Hash the components
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).digest()
    
    def _get_or_create_key(self) -> bytes:
        """Get existing key or create a new one"""
        if self._key:
            return self._key
        
        os.makedirs(os.path.dirname(self.key_file), exist_ok=True)
        
        if os.path.exists(self.key_file):
            # Load existing salt
            with open(self.key_file, 'rb') as f:
                salt = f.read()
        else:
            # Generate new salt
            salt = secrets.token_bytes(32)
            with open(self.key_file, 'wb') as f:
                f.write(salt)
            
            # Set restrictive permissions (Windows and Unix)
            try:
                os.chmod(self.key_file, 0o600)
            except:
                pass
        
        if CRYPTO_AVAILABLE:
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._get_machine_id()))
        else:
            # Simple key derivation fallback
            combined = salt + self._get_machine_id()
            key = base64.urlsafe_b64encode(hashlib.sha256(combined).digest())
        
        self._key = key
        return key
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a string value.
        
        Args:
            plaintext: String to encrypt
            
        Returns:
            Encrypted string (base64 encoded)
        """
        if not plaintext:
            return ""
        
        key = self._get_or_create_key()
        
        if CRYPTO_AVAILABLE:
            f = Fernet(key)
            encrypted = f.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        else:
            # Simple XOR obfuscation as fallback (NOT cryptographically secure!)
            # This is only for when cryptography package isn't available
            key_hash = hashlib.sha256(key).digest()
            encrypted = bytes(
                a ^ b for a, b in zip(
                    plaintext.encode(),
                    (key_hash * ((len(plaintext) // 32) + 1))[:len(plaintext)]
                )
            )
            return "WEAK:" + base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt an encrypted string.
        
        Args:
            ciphertext: Encrypted string from encrypt()
            
        Returns:
            Decrypted plaintext string
        """
        if not ciphertext:
            return ""
        
        key = self._get_or_create_key()
        
        if ciphertext.startswith("WEAK:"):
            # Handle weak encryption fallback
            ciphertext = ciphertext[5:]
            encrypted = base64.urlsafe_b64decode(ciphertext)
            key_hash = hashlib.sha256(key).digest()
            decrypted = bytes(
                a ^ b for a, b in zip(
                    encrypted,
                    (key_hash * ((len(encrypted) // 32) + 1))[:len(encrypted)]
                )
            )
            return decrypted.decode()
        
        if CRYPTO_AVAILABLE:
            try:
                encrypted = base64.urlsafe_b64decode(ciphertext)
                f = Fernet(key)
                return f.decrypt(encrypted).decode()
            except Exception as e:
                raise ValueError(f"Decryption failed: {e}")
        else:
            raise ValueError("Cannot decrypt: cryptography library not available")
    
    def is_encrypted(self, value: str) -> bool:
        """Check if a value appears to be encrypted"""
        if not value:
            return False
        
        # Check for our encryption markers
        if value.startswith("WEAK:"):
            return True
        
        # Check if it looks like Fernet output
        try:
            decoded = base64.urlsafe_b64decode(value)
            # Fernet tokens start with version byte
            return len(decoded) > 0
        except:
            return False


# Singleton instance
encryption_manager = EncryptionManager()
