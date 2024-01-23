from flask import Flask, request
from app.qna_unstructured.service import QnAService
from app.qna_structured.service import QnAStructuredService
from app.chat.service import ChatService

app = Flask(__name__)
app.debug = True
qnaservice_unstructured = QnAService()
qnaservice_structured=QnAStructuredService()
chatservice=ChatService()
# Members api route


@app.route("/getanswer")
def documentquery():
    query = request.args.get('query')
    answer = qnaservice_unstructured.get_answer(query)
    return answer

@app.route("/db")
def dbquery():
    query = request.args.get('query')
    answer = qnaservice_structured.get_answer(query)
    return answer

@app.route("/chat")
def chat():
    message = request.args.get('message')
    answer = chatservice.get_answer(message)
    return answer

if __name__ == "__main__":
    app.run(debug=True)
