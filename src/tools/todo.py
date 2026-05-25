import json
import os

TODO_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "todos.json")


def _load():
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(todos):
    os.makedirs(os.path.dirname(TODO_FILE), exist_ok=True)
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)


def create_todo(title: str, due_date: str = "") -> str:
    todos = _load()
    todo = {"id": len(todos) + 1, "title": title, "due_date": due_date, "done": False}
    todos.append(todo)
    _save(todos)
    due = f"，截止日期: {due_date}" if due_date else ""
    return f"待办已创建: #{todo['id']} {title}{due}"


def list_todos() -> str:
    todos = _load()
    if not todos:
        return "暂无待办事项"
    lines = ["当前待办事项:"]
    for t in todos:
        status = "✓" if t["done"] else "○"
        due = f" (截止: {t['due_date']})" if t.get("due_date") else ""
        lines.append(f"  {status} #{t['id']} {t['title']}{due}")
    return "\n".join(lines)
