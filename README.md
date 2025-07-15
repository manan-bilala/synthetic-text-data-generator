🧬 SynthGen – GPT-2 Powered Synthetic Text Data Generator

🚀 Project Overview

SynthGen is a no-code synthetic data generation platform powered by the (Flan-T5) model. It allows users to upload structured CSV datasets and generate synthetic records mimicking the original schema. Built using Streamlit and deployed via Ngrok, SynthGen is designed for data scientists, researchers, and developers who need high-quality artificial datasets.

💡 Key Features

- 🧠 Uses(`flan-t5-large`) for generating synthetic tabular text data
- 🧾 Accepts CSV uploads with up to 15 columns and at least 5 records
- 🛠️ Generates 50–200 synthetic records matching the structure
- 📉 Evaluates synthetic data against the original using similarity metrics
- 🎨 Beautiful, responsive Streamlit UI with multi-page navigation
- 🌐 One-click Ngrok deployment for sharing live demos


⚙️ How It Works

1. User uploads a CSV file (medical billing, transactional data, etc.)
2. SynthGen infers column types and sample examples
3. GPT-2 generates realistic synthetic records
4. Evaluation module compares real vs synthetic using:
   - Jaccard Similarity
   - Jensen–Shannon Divergence
   - Kolmogorov–Smirnov Test
   - Correlation Difference
5. Output is downloadable as a CSV

📁 Project Structure

```
synthetic-text-data-generator/
├── app.py                  # Streamlit app with UI and navigation
├── synthgen.py             # Main GPT-2 based generation logic
├── evaluator.py            # Evaluation logic for real vs synthetic
├── requirements.txt        # Python dependencies
├── sample_output.txt       # Example of synthetic text output
└── README.md               # Project documentation
```

📄 Sample Output

```
"Patient ID: 104; Procedure: Cardiac MRI; Amount: $1450.00; Department: Radiology; Insurance: Yes"
```

💻 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/synthetic-text-data-generator
cd synthetic-text-data-generator

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run app.py
```

🌐 Hosting with Ngrok

```bash
# Start your Streamlit app
streamlit run app.py

# In another terminal, run:
ngrok http 8501
```

🔮 Future Enhancements

- Prompt-based text generation
- Image generation using GANs
- Role-based access and login
- Multi-language synthetic generation

👨‍💻 Authors

- Dhruv Panchal
- Ronak Adep
- Manan Bilala


📜 License

This project is open-source and intended for academic and educational use only.
