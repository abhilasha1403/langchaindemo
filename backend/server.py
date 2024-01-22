from flask import Flask, request
from app.qna_unstructured.service import QnAService
from app.qna_structured.service import QnAStructuredService

app = Flask(__name__)
app.debug = True
qnaservice_unstructured = QnAService()
qnaservice_structured=QnAStructuredService()
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


if __name__ == "__main__":
    app.run(debug=True)
