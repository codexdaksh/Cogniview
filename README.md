🧠 Cogniview

Cogniview is a Streamlit-powered AI data analysis platform with a sleek glassmorphism interface.
It allows users to upload datasets, perform automated exploratory data analysis (EDA), and interact with their data using natural language queries powered by LLMs.

🚀 Features

📁 Smart Data Upload: Upload CSV/Excel files with instant metadata extraction

📊 Automated Analytics: EDA with statistical summaries, missing value detection, and interactive charts

🔮 Advanced Visualizations: Distributions, correlations, and relationship analysis

🧠 AI Assistant: Ask questions in natural language → get executable Pandas code and results

🎨 Premium UI: Glassmorphism design, smooth animations, and responsive layout

🛠️ Tech Stack

Streamlit
 – Web UI framework

Pandas
 – Data manipulation & processing

Matplotlib
 & Seaborn
 – Visualizations

LangChain
 – LLM orchestration

Ollama
 – Local LLM inference (configured with Mistral)

Python standard libraries: os, json, re

⚙️ Installation

Clone the repo:

git clone https://github.com/your-username/cogniview.git
cd cogniview


Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies:

pip install -r requirements.txt

▶️ Usage

Start the Ollama service (make sure ollama serve is running).
Example:

ollama run mistral


Launch the Streamlit app:

streamlit run cloud.py


Open your browser at http://localhost:8501 to explore Cogniview.

📂 Project Structure
├── cloud.py          # Main Streamlit app
├── requirements.txt  # Dependencies
├── metadata.json     # Generated automatically after dataset upload
└── README.md         # Documentation

📌 Notes

Ensure Ollama with Mistral model is installed and running.

Supports CSV and Excel datasets.

Automatically extracts metadata and provides analysis suggestions.

📝 License

This project is licensed under the MIT License.
Feel free to use, modify, and share.
