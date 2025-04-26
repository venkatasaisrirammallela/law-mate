# LawMate ⚖️ - AI-Powered Legal Advisor for Indian Laws

LawMate is an AI-driven legal assistant that provides accurate, context-aware answers to legal queries based on Indian laws. It processes uploaded PDFs and images, extracts legal information, and allows users to chat with an AI attorney.

---

## 🚀 Features

- Upload PDF and Image files (PNG, JPG, JPEG)
- Extracts text from documents and images
- Creates semantic search (vector database) using FAISS
- Uses Google's Gemini Model for answering queries
- Provides legal advice citing relevant Sections
- Maintains Chat History
- Automatically processes local dataset if no files are uploaded

---

## 🛠️ Tech Stack

- Python
- Streamlit
- LangChain
- FAISS (Vector Store)
- Google Generative AI (Gemini)
- Tesseract OCR
- PyPDF2
- PIL (Pillow)

---

## 📂 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
- Create a `.env` file in the root directory
- Add your Google API Key inside `.env`:

```
GOOGLE_API_KEY=your_google_api_key_here
```

(Or you can rename `.env.example` to `.env` and paste your key.)

---

## 🔑 .env.example
```plaintext
# Rename this file to ".env" and add your Google API Key
GOOGLE_API_KEY=your_google_api_key_here
```

---

## 📜 requirements.txt
```plaintext
streamlit
pytesseract
Pillow
PyPDF2
python-dotenv
langchain
langchain-google-genai
langchain-community
faiss-cpu
google-generativeai
```

---

## 🔧 Install Tesseract-OCR
- Download and Install Tesseract from:
  - [Tesseract OCR GitHub Releases](https://github.com/tesseract-ocr/tesseract)
- Update the path in your code if needed:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
```

---

## 🎯 Run the App
```bash
streamlit run app.py
```

---

## 📦 Folder Structure
```plaintext
your-project/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
└── dataset/   (optional, if you want to keep default PDFs)
```

---

## 📸 Screenshots

> (You can add screenshots of your app running here!)

---

## 📜 License

This project is licensed under the MIT License.

---

## ❤️ Acknowledgments

- Google Generative AI
- LangChain
- Streamlit Community
- Tesseract OCR Project

---

# ⚡ GitHub Commands to Push Project

```bash
# Initialize git
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit - LawMate AI Legal Advisor"

# Create GitHub repo (manually from GitHub website)

# Link local repo to remote GitHub repo
git remote add origin https://github.com/<your-username>/<your-repo-name>.git

# Push your code
git branch -M main
git push -u origin main
```
