# 🤟 Sign Language to Text & Speech Conversion System

A Machine Learning and Computer Vision based project that converts **sign language gestures into text and speech**.

This project aims to bridge the communication gap by recognizing sign language gestures through computer vision and translating them into readable text and audible speech output.

---

## ✨ Features

- 🤟 Sign language gesture recognition
- 📝 Sign to text conversion
- 🔊 Text to speech output
- 📷 Real time webcam based detection
- 🧠 Machine Learning powered recognition
- ⚡ Interactive and accessible communication system
- 🚀 Easy to run and extend

---

## 🛠️ Tech Stack

- Python
- OpenCV
- Machine Learning
- Computer Vision
- Speech Processing
- Text to Speech

---

## 📁 Project Structure

```text
sign-to-text-speech/
│
├── app.py
├── model.p
├── requirements.txt
├── README.md
├── LICENSE
│
├── static/
│   └── style.css
│
└── templates/
    └── index.html
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/kaushal033/sign-to-text-speech.git
cd sign-to-text-speech
```

### 2. Create and activate virtual environment (Recommended)

### Conda

```bash
conda create -n sign2speech python=3.11
conda activate sign2speech
```

### OR venv

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / Mac**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the main application:

```bash
python app.py
```

The system captures sign gestures through webcam input, processes them using the trained model, converts predictions into text, and generates speech output.

---

## 🧠 How It Works

1. Webcam captures hand gestures
2. Image preprocessing and feature extraction
3. Model predicts sign language gesture
4. Prediction converted into text
5. Text converted into speech output

This pipeline enables real time sign language interpretation and communication assistance.

---

## 📦 Requirements

All dependencies are listed inside:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Future Improvements

- Support for full sentence generation
- Improved gesture recognition accuracy
- Multi language speech output
- Mobile integration
- Larger sign vocabulary

---

## 🤝 Contributing

Contributions are welcome.

If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Commit and push
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

See the `LICENSE` file for more details.

---

## ⭐ Support

If you found this project useful, consider giving it a **star** on GitHub.

---

## 👨‍💻 Author

**Kaushal**

GitHub:  
https://github.com/kaushal033
