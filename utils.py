emoji_map = {
    "felicidade": "😄",
    "tristeza": "😢",
    "raiva": "😠",
    "medo": "😨",
    "amor": "🥰"
}

def get_emoji(label):
    return emoji_map.get(label.strip().lower(), "❓")