import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "model"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def predict_news(text):
    tokenizer, model = load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)[0]
    prediction = torch.argmax(probabilities).item()

    return {
        "label": "FAKE" if prediction == 1 else "REAL",
        "confidence": probabilities[prediction].item(),
        "real_probability": probabilities[0].item(),
        "fake_probability": probabilities[1].item(),
    }

if __name__ == "__main__":
    news = input("Enter news article: ")
    result = predict_news(news)
    print("Prediction:", result["label"])
    print("Confidence:", f'{result["confidence"] * 100:.2f}%')
