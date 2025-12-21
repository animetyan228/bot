from flask import Flask, request, jsonify
from flask_cors import CORS
from ai.pipeline import run_pipeline # главная ф-ция (crew ai + rag)

app = Flask(__name__)
app.json.ensure_ascii = False
CORS(app)

@app.route("/", methods=["POST"])
def chat():
    data = request.get_json()  # забираем запрос

    if not data or "message" not in data:
        return jsonify({"error": "Укажи message"})  # проверка на ошибку

    if not data.get("pdf_path"):
        return jsonify({"error": "Укажи pdf_path"})  # проверка на ошибку

    question = data["message"]  # вопрос пользователя
    pdf_path = data["pdf_path"]  # путь к пдф

    answer, context, evidence, error = run_pipeline(pdf_path, question) # запуск логики проекта
    if error:
        return jsonify({"error": error}) # если ошибка

    return jsonify({
    "answer": answer,
    "str": evidence,
    "chunk": context
}) # если все норм то ответ

# запускаем сервер
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)
