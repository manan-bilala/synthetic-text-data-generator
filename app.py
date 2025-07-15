!pip install streamlit pyngrok transformers pydantic torch pandas

from pyngrok import ngrok

# Optional: uncomment if you have a token from https://dashboard.ngrok.com/get-started/your-authtoken
ngrok.set_auth_token("2u81CRtOEeFPUSWmubaVdojWcML_5vaUx1nipJVGeFKSuYYC1")

%%writefile app.py
import streamlit as st
import pandas as pd
import tempfile
from synthgen import generate_synthetic_data
from evaluator import evaluate_data
from datetime import datetime

# Page config with modern theme
st.set_page_config(
    page_title="SYNTHGEN",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded"
)

# 🎨 Custom CSS (Enhanced UI)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700;900&display=swap');

    /* Main background */
    body, .stApp {
        background: linear-gradient(135deg, #6e8efb, #a777e3);
        font-family: 'Poppins', sans-serif;
        color: #ffffff;
        min-height: 100vh;
    }

    /* Main content container */
    .main {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        margin: 1rem;
    }

    /* Input fields */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: rgba(255, 255, 255, 0.9) !important;
        color: #2d3436 !important;
        border-radius: 12px !important;
        border: 2px solid #a6c1ee !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }

    /* Headings */
    .title {
        font-family: 'Montserrat', sans-serif;
        font-size: 3.5rem;
        text-align: center;
        background: linear-gradient(90deg, #ffffff, #a6c1ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }

    /* Buttons */
    .stButton>button {
        border-radius: 25px;
        background: linear-gradient(45deg, #6e8efb, #a777e3);
        color: white;
        border: none;
        padding: 14px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(110, 142, 251, 0.4);
    }

    /* Cards */
    .card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.18);
        transition: all 0.4s ease;
        height: 100%;
    }

    .card:hover {
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.15);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #6e8efb 0%, #a777e3 100%) !important;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        background: rgba(0, 0, 0, 0.2);
        color: white;
        margin-top: 3rem;
        border-radius: 15px;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .title {
            font-size: 2.5rem !important;
        }

        .main {
            padding: 1rem !important;
            margin: 0.5rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Custom Navigation Component
def custom_navigation():
    st.sidebar.markdown("""
        <div style="text-align: center; margin-bottom: 2rem;">
            <h1 style="font-family: 'Montserrat', sans-serif; color: white; font-size: 1.8rem;">
                <span style="color: #6e8efb;">SYNTH</span><span style="color: #a777e3;">GEN</span>
            </h1>
            <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem;">
                AI-Powered Data Solutions
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Initialize session state for page navigation
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "home"

    # Navigation items
    nav_items = [
        {"icon": "🏠", "label": "Home", "key": "home"},
        {"icon": "🔐", "label": "Login", "key": "login"},
        {"icon": "ℹ️", "label": "About", "key": "about"},
        {"icon": "📞", "label": "Contact", "key": "contact"}
    ]

    # Create navigation using buttons
    for item in nav_items:
        if st.sidebar.button(
            f"{item['icon']} {item['label']}",
            key=f"nav_{item['key']}",
            help=f"Go to {item['label']} page",
        ):
            st.session_state.selected_page = item["key"]

    return st.session_state.selected_page

# 🚀 Sidebar Navigation
selected_page = custom_navigation()

# 🏠 Home Page (Main Data Generation Functionality)
if selected_page == "home":
    st.markdown("<div class='title'>SYNTHETIC DATA GENERATOR</div>", unsafe_allow_html=True)

    with st.container():
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            uploaded_file = st.file_uploader("📤 Upload Medical Billing CSV", type="csv")
            num = st.slider("🎯 Number of Synthetic Records", 1, 50, 10)

        with col2:
            st.markdown("""
            <div class="card">
                <h3>📋 Upload Guidelines</h3>
                <ul style="color: rgba(255,255,255,0.8); line-height: 1.6">
                    <li>CSV format required</li>
                    <li>Max file size: 50MB</li>
                    <li>Structured column headers</li>
                    <li>No PII data allowed</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("### 📄 Uploaded Data Preview")
        st.dataframe(df.head().style.highlight_max(color='#f0abfc'))

        if st.button("🚀 Generate & Evaluate", use_container_width=True):
            with st.spinner("Generating synthetic records..."):
                synthetic_df = generate_synthetic_data(df, num)
                st.success("✅ Synthetic Data Generated!")
                st.dataframe(synthetic_df)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as orig_tmp, \
                 tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as synth_tmp:
                df.to_csv(orig_tmp.name, index=False)
                synthetic_df.to_csv(synth_tmp.name, index=False)

                st.subheader("📈 Evaluating Synthetic Data Quality...")
                similarity_scores, avg_score = evaluate_data(orig_tmp.name, synth_tmp.name)

                st.markdown(f"### 🧮 **Overall Similarity Score: `{avg_score}%`**")
                st.subheader("📊 Column-wise Scores:")
                for col, score in similarity_scores.items():
                    st.write(f"**{col}**: {score}%")

                st.download_button(
                    label="📥 Download Synthetic Data",
                    data=synthetic_df.to_csv(index=False),
                    file_name="synthetic_data.csv",
                    use_container_width=True
                )

# 🔐 Login Page
elif selected_page == "login":
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("""
            <div class="card" style="height: 300px; display: flex; justify-content: center; align-items: center;">
                <p style="color: rgba(255,255,255,0.5);">Security Illustration</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='title'>ACCOUNT LOGIN</div>", unsafe_allow_html=True)

        with st.form("login_form"):
            email = st.text_input("Email Address")
            password = st.text_input("Password", type="password")
            remember = st.checkbox("Remember me")

            if st.form_submit_button("Login", use_container_width=True):
                if email and password:
                    with st.spinner("Authenticating..."):
                        time.sleep(1.5)
                        st.success("Login successful!")
                else:
                    st.error("Please enter both email and password")

# ℹ️ About Page
elif selected_page == "about":
    st.markdown("<div class='title'>ABOUT US</div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="card">
            <h2>Our Mission</h2>
            <p style="color: rgba(255,255,255,0.8); line-height: 1.6;">
                We're revolutionizing how teams work with synthetic data. Our platform combines cutting-edge AI
                with an intuitive interface to deliver high-quality datasets that accelerate development while
                protecting privacy.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🧠 Technology")
    cols = st.columns(3)
    tech = [
        {"name": "AI Models", "desc": "Advanced generative models for realistic data"},
        {"name": "Privacy", "desc": "Differential privacy and anonymization"},
        {"name": "Scalability", "desc": "Cloud-native architecture for any workload"}
    ]

    for i, col in enumerate(cols):
        with col:
            st.markdown(f"""
                <div class="card">
                    <h3>{tech[i]['name']}</h3>
                    <p style="color: rgba(255,255,255,0.8);">{tech[i]['desc']}</p>
                </div>
            """, unsafe_allow_html=True)

# 📞 Contact Page
elif selected_page == "contact":
    st.markdown("<div class='title'>CONTACT US</div>", unsafe_allow_html=True)

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Message")

        if st.form_submit_button("Send Message", use_container_width=True):
            if name and email and message:
                st.success("Message sent! We'll get back to you soon.")
            else:
                st.warning("Please fill all fields")

    st.markdown("---")

    st.subheader("👥 Our Team")
    team_cols = st.columns(3)
    team_members = [
        {"name": "Dhruv Panchal", "role": "Student", "avatar": "👨‍💼", "bio": "Visionary leader with 10+ years in AI"},
        {"name": "Ronak Adep", "role": "Student", "avatar": "👩‍💻", "bio": "Tech innovator and systems architect"},
        {"name": "Manan Bilala", "role": "Student", "avatar": "👨‍🔬", "bio": "Machine learning expert"}
    ]

    for i, member in enumerate(team_members):
        with team_cols[i]:
            st.markdown(f"""
                <div class="card" style="text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">{member['avatar']}</div>
                    <h3>{member['name']}</h3>
                    <p style="color: #a777e3; font-weight: 600;">{member['role']}</p>
                    <p style="color: rgba(255,255,255,0.8);">{member['bio']}</p>
                </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
            <a href="#" style="color: white; text-decoration: none;">Privacy Policy</a>
            <a href="#" style="color: white; text-decoration: none;">Terms of Service</a>
            <a href="#" style="color: white; text-decoration: none;">Careers</a>
        </div>
        <p>© {datetime.now().year} SynthGen AI. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)

!pkill streamlit
!pkill ngro

!streamlit run app.py &> logs.txt &
from pyngrok import ngrok

public_url = ngrok.connect(8501)
print("🌍 Your app is live at:", public_url)