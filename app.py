import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from datetime import datetime
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


st.set_page_config(
    page_title="Система диагностики растений",
    page_icon="🌿",
    layout="wide"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/DenseNet121.keras")

model = load_model()

CLASS_NAMES = [
    "Перец: бактериальная пятнистость",
    "Перец: здоровый",
    "Картофель: ранняя фитофтора",
    "Картофель: поздняя фитофтора",
    "Картофель: здоровый",
    "Томат: бактериальная пятнистость",
    "Томат: ранняя фитофтора",
    "Томат: поздняя фитофтора",
    "Томат: листовая плесень",
    "Томат: септориоз",
    "Томат: паутинный клещ",
    "Томат: альтернариоз",
    "Томат: вирус желтой скручиваемости листьев",
    "Томат: вирус мозаики",
    "Томат: здоровый"
]


def save_history(pred_class, confidence):
    data = {
        "time": str(datetime.now()),
        "class": pred_class,
        "confidence": float(confidence)
    }

    file_path = "history.json"

    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []
    else:
        history = []

    history.append(data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=4)



def generate_pdf():
    if not os.path.exists("history.json"):
        return

    with open("history.json", "r", encoding="utf-8") as f:
        history = json.load(f)

    file_path = "report.pdf"
    c = canvas.Canvas(file_path, pagesize=A4)

    width, height = A4
    y = height - 50

    c.setFont("Helvetica", 11)
    c.drawString(50, y, "Plant Disease AI Report")
    y -= 30

    for item in history[-30:]:
        line = f"{item['time']} | {item['class']} | {item['confidence']:.2f}"
        c.drawString(50, y, line)
        y -= 20

        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)

    c.save()



st.title("🌿 Система диагностики заболеваний растений")
st.write("Загрузите изображение листа для анализа модели DenseNet121")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])

if uploaded_file:

    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.image(image, caption="Исходное изображение", use_container_width=True)

    
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)[0]

    class_id = int(np.argmax(pred))
    confidence = float(pred[class_id])

    save_history(CLASS_NAMES[class_id], confidence)

    with col2:
        st.subheader("Результат анализа")

        st.success(f"Класс: {CLASS_NAMES[class_id]}")

        st.metric("Уверенность", f"{confidence * 100:.2f}%")

        st.progress(confidence)

        st.subheader("Топ-3 вероятности")

        top3 = np.argsort(pred)[-3:][::-1]

        for i in top3:
            st.write(f"**{CLASS_NAMES[i]}** — {pred[i]*100:.2f}%")

        st.subheader("Полная статистика")

        st.bar_chart({CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))})


st.divider()

if st.button("📄 Скачать PDF отчёт"):
    generate_pdf()
    st.success("PDF отчёт создан")