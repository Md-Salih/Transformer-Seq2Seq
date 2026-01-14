# 🤖 Transformer Seq2Seq Text Summarization

**Production-Grade AI Summarization System** with Modern React UI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://reactjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

A production-ready text summarization system using the Transformer Encoder-Decoder architecture with **T5-Small** pre-trained model and a modern React + Vite frontend featuring real-time streaming generation.

### Key Features

✅ **Production-Ready Backend** - Flask API with streaming response support  
✅ **Pre-trained T5 Model** - Leverages Hugging Face's T5-Small for high-quality summaries  
✅ **Real-time Streaming** - Token-by-token generation with live progress tracking  
✅ **Modern React UI** - Split-screen layout with ChatGPT-inspired sidebar  
✅ **Statistics Dashboard** - Word count, reduction metrics, and AI detection percentage  
✅ **Clean Architecture** - Organized project structure with clear separation of concerns

---

## 🏗️ Project Structure

```
Transformer-Seq2Seq/
│
├── Backend Python Files
│   ├── app.py                    # Flask server with streaming API
│   ├── transformer.py            # Transformer model architecture
│   ├── encoder.py                # Encoder implementation
│   ├── decoder.py                # Decoder implementation
│   ├── attention_masks.py        # Attention mask utilities
│   ├── train.py                  # Training pipeline
│   ├── inference.py              # Inference utilities
│   ├── test_system.py            # System tests
│   └── requirements.txt          # Python dependencies
│
├── frontend/                      # React frontend
│   ├── src/
│   │   ├── App.jsx               # Main application
│   │   ├── App.css               # Application styles
│   │   ├── main.jsx              # React entry point
│   │   ├── index.css             # Global styles
│   │   └── components/
│   │       ├── Sidebar.jsx       # Navigation sidebar
│   │       └── Sidebar.css       # Sidebar styles
│   ├── package.json              # Node dependencies
│   └── vite.config.js            # Vite configuration
│
├── samples/                       # Example inputs
│   └── examples.md               # Sample text examples
│
├── .venv/                        # Python virtual environment
├── start.bat                     # Windows startup script
├── start.ps1                     # PowerShell startup script
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 18+**
- **pip** and **npm**

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Transformer-Seq2Seq
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # or
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install frontend dependencies:**
   ```bash
   cd frontend
   npm install
   cd ..
   ```

### Running the Application

**Option 1: Using the startup scripts (Windows)**
```bash
# Using batch file
start.bat

# Using PowerShell
.\start.ps1
```

**Option 2: Manual startup**

Terminal 1 - Backend:
```bash
.venv\Scripts\activate
python app.py
```

Terminal 2 - Frontend:
```bash
cd frontend
npm run dev
```

The application will be available at:
- **Frontend:** http://localhost:5173
- **Backend API:** http://localhost:5000

---

## 💡 Usage

1. **Enter Text:** Paste or type your text in the left input panel
2. **Click Summarize:** Press the white "Summarize" button to generate a summary
3. **Watch Live Progress:** See the white line loader fill up as the summary generates
4. **View Results:** Read the summary in the right output panel
5. **Check Stats:** Review word counts, reduction metrics, and AI detection percentage

### Example Texts

Click on the example items in the sidebar to quickly load sample texts:
- AI Technology trends
- Climate Change article
- Space Exploration piece

---

## 🛠️ Technology Stack

### Backend
- **Flask** - Lightweight web framework
- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Pre-trained T5-Small model
- **Python 3.8+** - Programming language

### Frontend
- **React 18** - UI library
- **Vite 5** - Build tool and dev server
- **CSS3** - Modern styling with ChatGPT-inspired theme

---

## 📊 API Endpoints

### `POST /api/summarize/stream`

Generates a summary with streaming response.

**Request:**
```json
{
  "text": "Your input text here..."
}
```

**Response (Server-Sent Events):**
```
data: {"token": "Summary", "progress": 10}
data: {"token": " begins", "progress": 25}
data: {"token": " here", "progress": 50}
data: {"done": true}
```

---

## 🎨 UI Features

### Split-Screen Layout
- **Left Panel:** Large textarea for input text with full scrolling support
- **Right Panel:** Summary display with real-time streaming output

### Progress Indicator
- **White Line Loader:** Smooth animated line that fills from 0-100% during generation
- **Percentage Display:** Shows exact progress percentage

### Statistics Dashboard
- **Input Words:** Total words in original text
- **Summary Words:** Total words in generated summary
- **Words Reduced:** Difference between input and summary
- **AI Detection:** Estimated AI content detection percentage

### Sidebar Navigation
- **New Chat Button:** Clear current session
- **Example Texts:** Quick-load predefined examples
- **User Profile:** Display current user

---

## 🧪 Testing

Run the test suite:
```bash
python test_system.py
```

---

## 📝 Configuration

### Model Settings (app.py)
- `model_name`: Default is "t5-small"
- `max_length`: Maximum summary length (default: 150)
- `min_length`: Minimum summary length (default: 40)

### Frontend Settings (vite.config.js)
- Dev server port: 5173
- API proxy: http://localhost:5000

---

## 🚧 Development

### Adding New Features

1. **Backend changes:** Modify `app.py` or model files
2. **Frontend changes:** Edit components in `frontend/src/`
3. **Styles:** Update CSS files for visual changes

### Code Style
- Backend: PEP 8 Python conventions
- Frontend: React best practices with functional components
- CSS: Modern CSS3 with custom properties

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **Hugging Face** - For the Transformers library and pre-trained models
- **React Team** - For the amazing React framework
- **Vite Team** - For the lightning-fast build tool

---

**Made with ❤️ using React + Vite + PyTorch**
