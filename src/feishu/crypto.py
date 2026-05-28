import base64
import hashlib
import struct
import string
import random
import json

from Crypto.Cipher import AES


class FeishuMsgCrypt:
    """飞书消息加解密"""

    def __init__(self, encrypt_key: str, app_id: str = ""):
        self.app_id = app_id
        # 飞书 Encrypt Key 需要先 base64 解码，再 SHA256 得到 32 字节 AES 密钥
        key_bytes = base64.b64decode(encrypt_key)
        if len(key_bytes) != 32:
            raise ValueError(
                f"Encrypt Key 解码后应为 32 字节，实际为 {len(key_bytes)} 字节"
            )
        self.aes_key = hashlib.sha256(key_bytes).digest()

    def _decrypt_msg(self, encrypt_text: str) -> bytes:
        """AES-256-CBC 解密，返回明文 bytes"""
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        plain_text = cipher.decrypt(base64.b64decode(encrypt_text))
        # PKCS#7 去填充
        pad = plain_text[-1]
        return plain_text[:-pad]

    def decrypt(self, encrypt_text: str) -> dict:
        """解密飞书加密消息，返回 JSON dict"""
        content = self._decrypt_msg(encrypt_text)
        # 格式: random(16) + msg_len(4, 网络字节序) + msg + app_id
        msg_len = struct.unpack("!I", content[16:20])[0]
        msg = content[20:20 + msg_len].decode("utf-8")
        return json.loads(msg)

    def encrypt(self, msg: str) -> str:
        """加密回复消息（飞书要求返回 JSON 字符串）"""
        random_chars = "".join(
            random.choices(string.ascii_letters + string.digits, k=16)
        )
        msg_bytes = msg.encode("utf-8")
        app_bytes = self.app_id.encode("utf-8")
        content = (
            random_chars.encode("utf-8")
            + struct.pack("!I", len(msg_bytes))
            + msg_bytes
            + app_bytes
        )
        # PKCS#7 填充到 32 字节倍数
        pad = 32 - len(content) % 32
        content += bytes([pad] * pad)
        cipher = AES.new(self.aes_key, AES.MODE_CBC, self.aes_key[:16])
        return base64.b64encode(cipher.encrypt(content)).decode()
