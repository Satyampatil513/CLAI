# CLAI: Command Line AI Assistant

CLAI is a smart, context-aware command line assistant for Windows. It can understand natural language prompts, generate safe and relevant commands, and learn from your history.

---

## 🚀 Quick Start (Recommended: Standalone Executable)

1. **Download the latest `clai.exe`** from the [GitHub Releases](https://github.com/Satyampatil513/CLAI/releases) page.
2. **Place `clai.exe` in a folder that is in your system PATH** (e.g., `C:\Windows` or create a `C:\Tools` folder and add it to PATH).
3. **Open a terminal (PowerShell or CMD) and run:**
   ```
   clai your prompt here
   ```
   Example:
   ```
   clai list all folders in this directory
   ```
4. **On first run,** you will be prompted for your Gemini API key. Enter it to create a `.env` file for future use.

---

## 🛠️ Developer Setup (from Source)

1. **Clone the repository:**
   ```
   git clone https://github.com/Satyampatil513/CLAI.git
   cd CLAI
   ```
2. **(Optional but recommended) Create a virtual environment:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```
4. **Set up your Gemini API key:**
   - On first run, you will be prompted for your API key, or you can create a `.env` file with:
     ```
     GEMINI_API_KEY=your_gemini_api_key_here
     ```
5. **Run the assistant:**
   ```
   python clai.py your prompt here
   ```

---

## 📝 Usage Notes
- You can run `clai` from any directory if `clai.exe` is in your PATH.
- The assistant will print all output in your terminal window.
- Dangerous commands (like `del`) will require confirmation.
- If you want to keep a log, use: `clai your prompt > output.txt`
- For best results, use clear, specific prompts.

---

## 🧠 How It Works
- CLAI uses local semantic memory (FAISS + sentence-transformers) and Google Gemini for command generation.
- All long-term memory is stored in `memory.sqlite` in the working directory.
- No data is sent anywhere except to Gemini (for command generation).

---

## ❓ Troubleshooting
- **Startup is slow?** The first run loads the AI model; subsequent runs are faster.
- **Network errors?** Check your internet connection and Gemini API key.
- **Permission errors?** Run your terminal as administrator if needed.

---

## 📦 Building Your Own Executable
- If you want to build `clai.exe` yourself:
   ```
   pip install pyinstaller
   pyinstaller --onefile --name clai clai.py
   ```
- The executable will be in the `dist` folder.

---

## 📄 License
MIT
