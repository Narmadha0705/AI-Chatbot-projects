import time
from collections import defaultdict, deque
 
from flask import Flask, jsonify, render_template, request
 
import chatbot
from search import search_knowledge
 
SITE_TITLE = "AVIT Assistant"          # change the name shown on the page here
HOST = "127.0.0.1"                     # "0.0.0.0" lets other devices on your Wi-Fi open it too
PORT = 5000
 
# simple protection for your Gemini quota: max requests per visitor per minute
RATE_LIMIT = 20
RATE_WINDOW = 60
_hits = defaultdict(deque)
 
app = Flask(__name__)
 
 
def too_many_requests(ip):
    now = time.time()
    hits = _hits[ip]
    while hits and now - hits[0] > RATE_WINDOW:
        hits.popleft()
    if len(hits) >= RATE_LIMIT:
        return True
    hits.append(now)
    return False
 
 
def clean_history(raw):
    """The browser sends the last few [question, answer] pairs; keep them small and safe."""
    history = []
    if isinstance(raw, list):
        for item in raw[-chatbot.HISTORY_TURNS:]:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                history.append((str(item[0])[:500], str(item[1])[:1500]))
    return history
 
 
@app.get("/")
def home():
    return render_template("index.html", title=SITE_TITLE, contact=chatbot.ADMISSIONS_CONTACT)
 
 
@app.post("/api/chat")
def chat():
    if too_many_requests(request.remote_addr):
        return jsonify(answer="You are asking too fast. Please wait a minute and try again."), 429
 
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()[:500]
    history = clean_history(data.get("history", []))
 
    if not question:
        return jsonify(answer="Please type a question."), 400
 
    try:
        quick = chatbot.quick_answer(question)
        if quick:
            return jsonify(answer=quick)
 
        results = search_knowledge(chatbot.make_search_query(question, history), top_k=chatbot.TOP_K)
 
        if not results or results[0]["score"] < chatbot.MIN_SCORE:
            return jsonify(answer="I couldn't find that in the university information. "
                                  f"Please contact admissions: {chatbot.ADMISSIONS_CONTACT}")
 
        answer = chatbot.generate_answer(question, results, history, raw_fallback=False)
        return jsonify(answer=answer + chatbot.rules_note(question))
 
    except Exception as error:
        print("Error while answering:", error)
        return jsonify(answer=chatbot.BUSY_MESSAGE), 500
 
 
if __name__ == "__main__":
    print("Loading the knowledge base (first start takes a few seconds)...")
    search_knowledge("warm up", top_k=1)      # so the first visitor doesn't wait
    print(f"\nWebsite is running:  http://{'127.0.0.1' if HOST == '0.0.0.0' else HOST}:{PORT}\n")
    app.run(host=HOST, port=PORT, debug=False)
 