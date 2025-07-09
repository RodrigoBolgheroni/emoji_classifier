emoji_map = {
    "alegria": "😄",
    "tristeza": "😢",
    "raiva": "😠",
    "medo": "😨",
    "amor": "🥰"
}

def get_emoji(label):
    return emoji_map.get(label, "❓")