def send_notification(recipient: str, message: str) -> str:
    return f"[IM通知模拟] 发送给: {recipient}, 消息: {message[:100]}{'...' if len(message) > 100 else ''}"
