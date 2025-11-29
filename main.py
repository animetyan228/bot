from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import os

# ===============================
# 1. Инициализация Flask и OpenAI
# ===============================

app = Flask(__name__)
CORS(app)

# Получаем ключ API из переменной окружения
# Обязательно добавь переменную окружения OPENAI_API_KEY перед запуском
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Не найдена переменная окружения OPENAI_API_KEY!")

client = OpenAI(api_key=api_key)

# ===============================
# 2. Endpoint /chat
# ===============================

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "message" not in data:
        return jsonify({"error": "Поле 'message' отсутствует"}), 400

    user_msg = data["message"]

    try:
        resp = client.responses.create(
            model="gpt-5-nano",  # Можно заменить на gpt-4o-mini
            input=user_msg
        )
        answer = resp.output_text
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ===============================
# 3. Запуск сервера
# ===============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
