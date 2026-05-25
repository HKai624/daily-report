def send_email(to: str, subject: str, body: str) -> str:
    return f"[邮件模拟] 收件人: {to}, 主题: {subject}, 内容: {body[:100]}{'...' if len(body) > 100 else ''}"
