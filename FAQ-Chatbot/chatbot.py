import tkinter as tk
from tkinter import scrolledtext, messagebox
from datetime import datetime
import pandas as pd
import nltk
import string

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass

faq_data = pd.read_csv("faq.csv")

questions = faq_data["Question"].tolist()
answers = faq_data["Answer"].tolist()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return " ".join(tokens)

processed_questions = [preprocess_text(q) for q in questions]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(processed_questions)

def get_best_answer(user_question):

    processed_question = preprocess_text(user_question)

    user_vector = vectorizer.transform([processed_question])

    similarity_scores = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarity_scores.argmax()

    best_score = similarity_scores[0][best_match_index]

    if best_score < 0.35:
        return (
            "Sorry, I couldn't find a suitable answer. "
            "Please try asking your question in a different way."
        )

    return answers[best_match_index]


def send_message():

    user_question = entry.get().strip()

    if user_question == "":
        return

    answer = get_best_answer(user_question)

    current_time = datetime.now().strftime("%I:%M %p")

    chat_area.config(state=tk.NORMAL)

    chat_area.insert(
        tk.END,
        f"\n[{current_time}] You: {user_question}\n"
    )

    chat_area.insert(
        tk.END,
        f"[{current_time}] Bot: {answer}\n\n"
    )

    chat_area.config(state=tk.DISABLED)

    chat_area.see(tk.END)

    entry.delete(0, tk.END)

def clear_chat():

    chat_area.config(state=tk.NORMAL)

    chat_area.delete(1.0, tk.END)

    chat_area.insert(
        tk.END,
        "AI Learning Assistant\n\n"
        "Hello! Ask me anything about AI, Python or Machine Learning.\n\n"
    )

    chat_area.config(state=tk.DISABLED)

def show_about():

    messagebox.showinfo(
        "About",
        "AI Learning Assistant (FAQ Chatbot)\n\n"
        "Developed by: Shruti Singh\n\n"
        "Technologies Used:\n"
        "• Python\n"
        "• Tkinter\n"
        "• Pandas\n"
        "• NLTK\n"
        "• Scikit-learn\n"
        "• TF-IDF\n"
        "• Cosine Similarity\n\n"
        "CodeAlpha AI Internship Project"
    )

root = tk.Tk()

root.title("AI Learning Assistant")
root.geometry("700x600")
root.configure(bg="#F4F6F8")

title_label = tk.Label(
    root,
    text="AI Learning Assistant",
    font=("Arial", 20, "bold"),
    bg="#F4F6F8",
    fg="#1F4E79"
)

title_label.pack(pady=10)

subtitle = tk.Label(
    root,
    text="Ask questions related to Artificial Intelligence, Python and Machine Learning",
    font=("Arial", 10),
    bg="#F4F6F8"
)

subtitle.pack()

chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    width=75,
    height=22,
    font=("Calibri", 11),
    state=tk.NORMAL
)

chat_area.pack(padx=15, pady=10)

chat_area.insert(
    tk.END,
    "AI Learning Assistant\n\n"
    "Hello! Welcome.\n"
    "Ask me anything related to AI, Python or Machine Learning.\n\n"
)

chat_area.config(state=tk.DISABLED)

bottom_frame = tk.Frame(root, bg="#F4F6F8")

bottom_frame.pack(fill=tk.X, padx=10, pady=10)

entry = tk.Entry(
    bottom_frame,
    font=("Arial", 12),
    width=48
)

entry.pack(side=tk.LEFT, padx=5)

send_button = tk.Button(
    bottom_frame,
    text="Send",
    font=("Arial", 11, "bold"),
    bg="#1F4E79",
    fg="white",
    width=10,
    command=send_message
)

send_button.pack(side=tk.LEFT, padx=5)

clear_button = tk.Button(
    bottom_frame,
    text="Clear Chat",
    font=("Arial", 11, "bold"),
    bg="#C0392B",
    fg="white",
    width=10,
    command=clear_chat
)

clear_button.pack(side=tk.LEFT, padx=5)

about_button = tk.Button(
    bottom_frame,
    text="About",
    font=("Arial", 11, "bold"),
    bg="#27AE60",
    fg="white",
    width=10,
    command=show_about
)

about_button.pack(side=tk.LEFT, padx=5)

exit_button = tk.Button(
    bottom_frame,
    text="Exit",
    font=("Arial", 11, "bold"),
    bg="#555555",
    fg="white",
    width=10,
    command=root.destroy
)

exit_button.pack(side=tk.LEFT, padx=5)

entry.bind("<Return>", lambda event: send_message())

root.mainloop()

