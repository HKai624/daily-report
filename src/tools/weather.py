import requests


def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=%C+%t+%h+%w&lang=zh"
        resp = requests.get(url, timeout=10, headers={"User-Agent": "curl"})
        if resp.status_code == 200:
            return f"{city}天气: {resp.text.strip()}"
        return f"无法获取{city}的天气信息(HTTP {resp.status_code})"
    except Exception as e:
        return f"天气查询失败: {str(e)}"
