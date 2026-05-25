"""Debug Gradio 6.0 chat bubble CSS class names."""
import requests, re

r = requests.get("http://127.0.0.1:8000/", timeout=5)
text = r.text

# Find all class names containing chat or message or bubble
for pattern in ["bubble", "message", "chat", "user", "bot", "assistant"]:
    found = set(re.findall(rf'class="([^"]*{pattern}[^"]*)"', text))
    if found:
        print(f"[{pattern}]: {list(found)[:8]}")
    else:
        print(f"[{pattern}]: NOT FOUND")

# Look for JSX/React-style class names
print("\n--- Looking for Gradio-generated chat classes ---")
matches = re.findall(r'class="([^"]*)"', text)
chat_classes = [m for m in matches if any(kw in m.lower() for kw in ["chat","bubble","msg","conversation"])]
for c in sorted(set(chat_classes))[:20]:
    print(f"  {c}")

print("\n--- Gradio JS bundle URLs ---")
js_files = re.findall(r'src="([^"]*\.js[^"]*)"', text)
for j in js_files[:5]:
    print(f"  {j[:100]}")
