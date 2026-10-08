"""
University chatbot = search (retrieval) + LLM (answer writing).

The best chunks are given to Gemini, which writes a short, precise
answer using ONLY that text (and says so when the answer isn't there).
The last few questions are remembered, so follow-ups like "is this high?" work.

Setup (Windows):
    pip install google-genai
    setx GEMINI_API_KEY "your-key-here"      (then open a NEW terminal)

Without an API key, or if Gemini is overloaded, the bot prints the top chunks.
"""

import os
import re
import time

from search import search_knowledge, REGULATION_WORDS, MODE_PATTERNS

# ==========================================
# SETTINGS
# ==========================================

TOP_K = 8
MIN_SCORE = 0.40          # below this the question is probably not covered. Tune it:
                          # set DEBUG = True and look at the printed scores.
DEBUG = False

# Tried in order. If the first is overloaded (503) or missing (404) the next one is used.
# Use the model names shown in Google AI Studio if these are ever retired.
MODELS = ["gemini-3.8-flash", "gemini-3-flash-preview"]
RETRIES_PER_MODEL = 3

HISTORY_TURNS = 3         # how many previous questions the bot remembers

ADMISSIONS_CONTACT = "080 150 111 56 / admissions@avit.ac.in"

# Shown on the website when Gemini cannot answer (the website never shows raw passages)
BUSY_MESSAGE = (
    "Sorry, I can't answer right now. Please try again in a minute, "
    f"or contact admissions: {ADMISSIONS_CONTACT}"
)

SYSTEM_PROMPT = f"""You are the official assistant for AVIT (AVIT / Vinayaka Mission's Research Foundation).
Answer the student's question using ONLY the context passages provided.

Rules:
- Be precise and short. Give exact numbers, dates and names as written in the context.
- For fees, rules or dates, always mention the programme and the academic year / regulation year they belong to.
- If the context has conflicting years or programmes, say which is which instead of guessing.
- Passages whose source is a web address (https://avit.ac.in/...) are the current official website. Prefer them
  over regulation PDFs for admission, fees, hostel and general information. Use the regulation PDFs for rules
  such as attendance, examinations, grading and PhD regulations. If a regulation passage is only about part-time,
  working-professional or PhD programmes, say so instead of presenting it as the rule for regular UG students.
- If the programme or topic the student asks about is NOT in the context (for example a fee for a programme that
  is not listed), say clearly that you don't have it. Do NOT give the fees of other programmes instead.
- Never include bank account numbers or bank details in an answer.
- If the answer is not in the context, say you don't have that information and suggest contacting
  the admissions office ({ADMISSIONS_CONTACT}). Never invent facts.
- If the student doesn't say which programme or mode (full-time, part-time, working professional, PhD), answer for
  regular full-time UG (B.E./B.Tech) using the latest regulation year. Do not list every programme's rules.
  Never write a sentence saying that PG, part-time or working-professional rules differ; the program adds it
  when it is needed.
- For fee questions, treat B.E. and B.Tech as the same level: if the student asks about "B.Tech" or "B.E." fees
  without naming a branch, give a compact list of every programme and its annual tuition fee for the latest
  academic year in the context, mention that the fee depends on 10+2 marks or VMRF-SMART rank, and list the
  extra fees (career advancement, caution deposit, skills academy) once. Do not mix fee tables of different
  programme groups into one programme's fee.
- When a website passage answers the question, do not add details about other modes from regulation PDFs.
- The previous conversation is given only so you understand words like "this", "it" or "that". Answer the
  current question about that same topic, not about a different one.
- If the student asks for an opinion (is it high, cheap, worth it, which is better), do not give a personal
  judgement. In at most 5 lines, give the lowest and highest figures and the scholarship range found in the
  context, so the student can decide. Do not repeat the full list from the previous answer.
- Answer only what was asked. Do not add department highlights, patents, research or other extra information.
- Write amounts in Indian format (for example ₹4,23,780). When listing fees or room types, include every row found
  in the context; do not skip any.
- At the end, add one line: "Source: ..." naming the documents or web addresses you used, without the
  "[Source: ...]" wrapper. Skip this line when you say you don't have the information."""


# ==========================================
# FOLLOW-UP HANDLING
# ==========================================

def is_followup(question):
    """Short questions or ones with 'this/that/it' depend on the previous question."""
    q = question.lower()
    if len(q.split()) <= 6:
        return True
    return bool(re.search(r"\b(this|that|it|its|those|these|they|them|same)\b", q))


def make_search_query(question, history):
    """For a follow-up, search with the last two questions as well, so the topic is not lost."""
    if history and is_followup(question):
        earlier = " ".join(q for q, _ in history[-2:])
        return earlier + " " + question
    return question


def history_text(history):
    if not history:
        return ""
    lines = ["Previous conversation (only to understand what the student means):"]
    for q, a in history:
        lines.append(f"Student: {q}")
        lines.append(f"Assistant: {a[:600]}")
    return "\n".join(lines) + "\n\n"


# ==========================================
# ANSWER GENERATION
# ==========================================

def build_context(results):
    return "\n\n---\n\n".join(r["text"] for r in results)


def show_passages(results, note):
    lines = [note + "\n"]
    for r in results[:3]:
        lines.append(r["text"][:900])
        lines.append("-" * 60)
    return "\n".join(lines)


def generate_answer(question, results, history, raw_fallback=True):
    """raw_fallback=True (terminal): show passages if Gemini fails. False (website): friendly message."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        if not raw_fallback:
            return BUSY_MESSAGE
        return show_passages(results, "(No GEMINI_API_KEY set - showing the best matching passages)")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    contents = (
        f"{history_text(history)}"
        f"Context passages:\n\n{build_context(results)}\n\n"
        f"Student question: {question}"
    )
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_error = None
    for model in MODELS:
        for attempt in range(RETRIES_PER_MODEL):
            try:
                response = client.models.generate_content(model=model, contents=contents, config=config)
                if response.text:
                    return response.text
                last_error = "empty answer"
            except Exception as error:
                last_error = error
                message = str(error)
                if "404" in message or "NOT_FOUND" in message:
                    break                       # model not available -> try the next model
                if DEBUG:
                    print(f"   [debug] {model} attempt {attempt + 1} failed: {message[:120]}")
                time.sleep(2 * (attempt + 1))   # overloaded / rate limit -> wait and retry

    print(f"[Gemini failed] {str(last_error)[:300]}")   # the real reason shows in the terminal

    if not raw_fallback:
        return BUSY_MESSAGE

    return show_passages(
        results,
        f"(Gemini is busy or unavailable right now: {str(last_error)[:150]})\n"
        "Showing the best matching passages instead:",
    )


# ==========================================
# QUICK ANSWERS (no search needed)
# ==========================================

def quick_answer(question):
    q = question.lower()
    asks_contact = any(w in q for w in ("contact", "phone", "email", "e-mail", "call", "helpline"))
    if asks_contact and "admission" in q:
        return f"For admissions, contact: {ADMISSIONS_CONTACT}"
    return None


def rules_note(question):
    """Added only to rule questions (attendance, exams ...) that don't name a programme mode."""
    q = question.lower()
    is_rule_question = any(word in q for word in REGULATION_WORDS)
    names_a_mode = any(re.search(pattern, q) for pattern, _ in MODE_PATTERNS)
    if is_rule_question and not names_a_mode:
        return "\n\nNote: PG, part-time and working-professional rules differ."
    return ""


# ==========================================
# CHAT LOOP
# ==========================================

def main():
    print("=" * 70)
    print("AI UNIVERSITY CHATBOT")
    print("=" * 70)
    print("Ask your question (type exit to stop)")

    history = []     # [(question, answer), ...]

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("Bot: Goodbye!")
            break
        if not question:
            print("Bot: Please enter a question.")
            continue

        quick = quick_answer(question)
        if quick:
            print("\nBot:", quick)
            continue

        results = search_knowledge(make_search_query(question, history), top_k=TOP_K)

        if DEBUG:
            for r in results:
                print(f"   [debug] score={r['score']:.3f}  {r['text'][:70]!r}")

        if not results or results[0]["score"] < MIN_SCORE:
            print("\nBot: I couldn't find that in the university information. "
                  f"Please contact admissions: {ADMISSIONS_CONTACT}")
            continue

        answer = generate_answer(question, results, history)
        print("\nBot:", answer + rules_note(question))

        history.append((question, answer))
        history = history[-HISTORY_TURNS:]


if __name__ == "__main__":
    main()