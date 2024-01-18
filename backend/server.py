from flask import Flask , request
from app.qna.service import QnAService

app=Flask(__name__)

qnaservice=QnAService()
#Members api route
@app.route("/getanswer")
def members():
    query = request.args.get('param')
    answer=qnaservice.get_answer(query)
    return answer

if __name__=="__main__":
    app.run(debug=True)