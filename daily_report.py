import os
import sys
import smtplib
import time
from email.mime.text import MIMEText
from email.utils import formataddr
from datetime import datetime

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

QQ_ADDRESS = os.getenv("QQ_EMAIL_ADDRESS", "")
QQ_PASSWORD = os.getenv("QQ_EMAIL_PASSWORD", "")
RECIPIENT = os.getenv("QQ_EMAIL_ADDRESS", "")

# 南宁经纬度
NANNING_LAT = 22.82
NANNING_LON = 108.37

# Open-Meteo WMO 天气码 → 中文
WMO_CODES = {
    0: "晴天", 1: "少云", 2: "多云", 3: "阴天",
    45: "雾", 48: "雾凇",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "小冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪",
    77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "阵雪",
    95: "雷暴", 96: "雷暴+小冰雹", 99: "雷暴+大冰雹",
}


def get_nanning_weather():
    """Open-Meteo — 南宁实时天气（免费，无需 Key）"""
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={NANNING_LAT}&longitude={NANNING_LON}"
        f"&current=temperature_2m,weather_code,relative_humidity_2m,wind_speed_10m"
        f"&timezone=Asia/Shanghai"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        cur = data.get("current", {})
        temp = cur.get("temperature_2m", "?")
        code = cur.get("weather_code", -1)
        humidity = cur.get("relative_humidity_2m", "?")
        wind = cur.get("wind_speed_10m", "?")
        text = WMO_CODES.get(code, f"未知({code})")
        return f"{text}，{temp}℃，湿度{humidity}%，风速{wind}km/h"
    except Exception as e:
        return f"天气查询异常: {e}"


def get_daily_news():
    """60s API — 每日新闻（免费，无需 Key）"""
    url = "https://60s.viki.moe/v2/60s"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") == 200:
            items = data["data"].get("news", [])
            return items[:8] if len(items) >= 8 else items
        return []
    except Exception:
        return []


def format_email_content(weather_str, news_list, date_str):
    if news_list:
        news_block = "\n".join(
            [f"  {i+1}. {item}" for i, item in enumerate(news_list)]
        )
    else:
        news_block = "  （暂未获取到新闻）"

    return (
        f"早安！\n\n"
        f"{date_str}\n\n"
        f"南宁今日天气：{weather_str}\n\n"
        f"今日热点：\n{news_block}\n\n"
        f"---\n"
        f"本邮件由 AI 智能助手自动发送。"
    )


def send_email(content, to_addr):
    smtp_server = "smtp.qq.com"
    port = 587

    msg = MIMEText(content, "plain", "utf-8")
    msg["From"] = formataddr(("AI智能助手", QQ_ADDRESS))
    msg["To"] = formataddr(("你", to_addr))
    msg["Subject"] = f"每日南宁天气与新闻 ({datetime.now().strftime('%m-%d')})"

    try:
        server = smtplib.SMTP(smtp_server, port, timeout=15)
        server.starttls()
        server.login(QQ_ADDRESS, QQ_PASSWORD)
        server.sendmail(QQ_ADDRESS, [to_addr], msg.as_string())
        print(f"[{datetime.now():%H:%M:%S}] 邮件发送成功 -> {to_addr}")
    except Exception as e:
        print(f"[{datetime.now():%H:%M:%S}] 发送失败: {e}")
    finally:
        try:
            server.quit()
        except Exception:
            pass


def daily_job():
    print(f"[{datetime.now():%H:%M:%S}] 执行每日推送...")
    weather = get_nanning_weather()
    news = get_daily_news()
    today = datetime.now().strftime("%Y年%m月%d日")
    content = format_email_content(weather, news, today)
    send_email(content, RECIPIENT)
    print(f"  天气: {weather}")
    print(f"  新闻: {len(news)} 条")


def main():
    if not QQ_ADDRESS or QQ_ADDRESS == "your_qq@qq.com":
        print("[WARN] 未设置 QQ_EMAIL_ADDRESS")
        sys.exit(1)
    if not QQ_PASSWORD:
        print("[WARN] 未设置 QQ_EMAIL_PASSWORD")
        sys.exit(1)

    is_daemon = len(sys.argv) > 1 and sys.argv[1] == "--daemon"

    if is_daemon:
        import schedule

        schedule.every().day.at("08:30").do(daily_job)
        print("定时器已启动，每天 08:30 执行推送...")
        print("按 Ctrl+C 停止")
        while True:
            schedule.run_pending()
            time.sleep(60)
    else:
        print(f"发件人: {QQ_ADDRESS}")
        print(f"收件人: {RECIPIENT}")
        weather = get_nanning_weather()
        print(f"天气: {weather}")
        news = get_daily_news()
        for i, n in enumerate(news, 1):
            print(f"新闻{i}: {n}")
        daily_job()


if __name__ == "__main__":
    main()
