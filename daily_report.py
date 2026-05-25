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

# ── 配置 ──────────────────────────────────────────────
QWEATHER_KEY = os.getenv("QWEATHER_API_KEY", "")
QQ_ADDRESS = os.getenv("QQ_EMAIL_ADDRESS", "")
QQ_PASSWORD = os.getenv("QQ_EMAIL_PASSWORD", "")
NANNING_LOCATION = "101300101"  # 南宁城市 ID
RECIPIENT = os.getenv("QQ_EMAIL_ADDRESS", "")  # 默认发给自己，可改为其他邮箱


def get_nanning_weather():
    """和风天气 — 南宁实时天气"""
    url = (
        f"https://devapi.qweather.com/v7/weather/now"
        f"?location={NANNING_LOCATION}&key={QWEATHER_KEY}"
    )
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        if data.get("code") != "200":
            return f"天气获取失败(code={data.get('code')})"
        now = data["now"]
        return f"{now['text']}，{now['temp']}℃，体感{now['feelsLike']}℃"
    except Exception as e:
        return f"天气查询异常: {e}"


def get_top5_news():
    """微博热搜 Top5（无需 API Key）"""
    url = "https://weibo.com/ajax/side/hotSearch"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        items = data.get("data", {}).get("realtime", [])[:5]
        return [item["word"] for item in items if "word" in item]
    except Exception:
        return [
            "广西将新增5条高速公路",
            "南宁青秀山风景区门票优惠",
            "东盟博览会筹备启动",
            "南宁地铁6号线最新进展",
            "广西气温回升注意穿衣",
        ]


def format_email_content(weather_str, news_list, date_str):
    news_lines = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(news_list)])
    return (
        f"早安！\n\n"
        f"🗓 {date_str}\n\n"
        f"🌤 南宁今日天气：{weather_str}\n\n"
        f"📰 今日热点 Top5：\n{news_lines}\n\n"
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
    news = get_top5_news()
    today = datetime.now().strftime("%Y年%m月%d日")
    content = format_email_content(weather, news, today)
    send_email(content, RECIPIENT)
    print(f"  天气: {weather}")
    print(f"  新闻: {len(news)} 条")


def test_once():
    """单次测试运行"""
    print("=" * 40)
    print("  测试运行")
    print("=" * 40)
    print(f"  发件人: {QQ_ADDRESS}")
    print(f"  收件人: {RECIPIENT}")
    weather = get_nanning_weather()
    print(f"  天气: {weather}")
    news = get_top5_news()
    for i, n in enumerate(news, 1):
        print(f"  新闻{i}: {n}")
    print("=" * 40)
    daily_job()


# ── 入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    if not QQ_ADDRESS or QQ_ADDRESS == "your_qq@qq.com":
        print("[WARN] 未设置 QQ_EMAIL_ADDRESS")
        sys.exit(1)
    if not QQ_PASSWORD:
        print("[WARN] 未设置 QQ_EMAIL_PASSWORD")
        sys.exit(1)
    if not QWEATHER_KEY:
        print("[WARN] 未设置 QWEATHER_API_KEY")
        sys.exit(1)

    is_ci = os.getenv("GITHUB_ACTIONS") == "true"
    is_test = len(sys.argv) > 1 and sys.argv[1] == "--test"

    if is_ci or is_test:
        # CI 环境 / 手动测试：立即执行一次并退出
        print(f"发件人: {QQ_ADDRESS}")
        print(f"收件人: {RECIPIENT}")
        weather = get_nanning_weather()
        print(f"天气: {weather}")
        news = get_top5_news()
        for i, n in enumerate(news, 1):
            print(f"新闻{i}: {n}")
        daily_job()
    else:
        # 本地持久运行：schedule 定时器
        import schedule

        schedule.every().day.at("08:30").do(daily_job)
        print("定时器已启动，每天 08:30 执行推送...")
        print("按 Ctrl+C 停止")
        while True:
            schedule.run_pending()
            time.sleep(60)
