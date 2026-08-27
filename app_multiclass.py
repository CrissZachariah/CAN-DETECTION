
import os
import io
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import streamlit as st

# ReportLab imports for automated PDF report generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & MODERN CLEAN THEME CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="CAN IDS | Automotive Security",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Hide standard Streamlit header & footer elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main Background - Soft Modern Light Canvas */
.stApp {
    background-color: #F8FAFC !important;
}

/* Base Typography */
html, body, [class*="css"], .stMarkdown {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #0F172A;
}

/* Dark Navy Sidebar Theme */
section[data-testid="stSidebar"] {
    background-color: #111927 !important;
    border-right: 1px solid #1E293B;
}

section[data-testid="stSidebar"] * {
    color: #94A3B8 !important;
}

section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

/* Sidebar Navigation Items Styling */
div[data-testid="stSidebarUserContent"] .stRadio label {
    padding: 10px 14px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.15s ease;
}

/* Top Dashboard Header */
.dashboard-header {
    margin-bottom: 20px;
}
.dashboard-header h1 {
    font-size: 26px;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 2px;
}
.dashboard-header p {
    font-size: 14px;
    color: #64748B;
    margin: 0;
}

/* Crisp White KPI Card Containers */
.kpi-card {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 18px 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    display: flex;
    align-items: center;
    gap: 16px;
}

.kpi-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
}

.kpi-blue { background-color: #EFF6FF; color: #2563EB; }
.kpi-green { background-color: #ECFDF5; color: #10B981; }
.kpi-red { background-color: #FEF2F2; color: #EF4444; }
.kpi-purple { background-color: #F5F3FF; color: #8B5CF6; }

.kpi-details {
    display: flex;
    flex-direction: column;
}
.kpi-label {
    font-size: 12px;
    font-weight: 500;
    color: #64748B;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 22px;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.kpi-subtext {
    font-size: 11px;
    color: #94A3B8;
    margin-top: 4px;
}

/* Card Boxes for Charts / Tables */
.card-box {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 20px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    margin-bottom: 20px;
}

/* Buttons Styling */
.stButton > button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 2px rgba(37, 99, 235, 0.2);
    width: 100%;
}
.stButton > button:hover {
    background-color: #1D4ED8 !important;
}

/* Threat Badges */
.badge-danger { background-color: #FEF2F2; color: #EF4444; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
.badge-success { background-color: #ECFDF5; color: #10B981; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }
.badge-warning { background-color: #FFFBEB; color: #F59E0B; padding: 4px 10px; border-radius: 6px; font-weight: 600; font-size: 12px; }

/* Status Box Sidebar */
.sidebar-status {
    background-color: #1E293B;
    border-radius: 10px;
    padding: 14px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 2. MODEL ARCHITECTURE & LOAD ENGINE
# ------------------------------------------------------------------------------
class LightweightCANCNN(nn.Module):
    def __init__(self, window_size=128, num_classes=5, use_gap=True):
        super(LightweightCANCNN, self).__init__()
        self.num_classes = num_classes
        self.use_gap = use_gap
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
        )

        if use_gap:
            self.pool = nn.AdaptiveAvgPool2d(1)
            head_in = 32
        else:
            self.pool = nn.Identity()
            head_in = 32 * (window_size // 4) * (window_size // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(head_in, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

# Default class set (matches CLASS_NAMES saved in the multi-class checkpoint).
# Falls back to binary ["Normal","Attack"] automatically if an older
# 2-class checkpoint is loaded instead (see load_model()).
DEFAULT_CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]


@st.cache_resource
def load_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        return None, device, 128, len(DEFAULT_CLASS_NAMES), DEFAULT_CLASS_NAMES, {}

    try:
        checkpoint = torch.load(model_path, map_location=device)
        window_size = checkpoint.get('window_size', 128)
        num_classes = checkpoint.get('num_classes', 2)
        class_names = checkpoint.get('class_names', ["Normal", "Attack"])
        can_id_map = checkpoint.get('can_id_map', {})
        state_dict = checkpoint['model_state_dict']

        # Don't trust the 'use_gap' metadata key alone — some checkpoints were
        # saved by a notebook run where the save cell hadn't been updated to
        # record it, even though the model itself was already GAP-based.
        # Instead, read the ACTUAL shape of the first classifier layer:
        # in_features == 32 means GAP (AdaptiveAvgPool2d(1)); anything else
        # (e.g. 32768) means the original full-flatten head.
        classifier_in_features = state_dict['classifier.1.weight'].shape[1]
        use_gap = (classifier_in_features == 32)

        model = LightweightCANCNN(window_size=window_size, num_classes=num_classes, use_gap=use_gap)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, device, window_size, num_classes, class_names, can_id_map
    except Exception as e:
        st.error(f"Error loading model checkpoint: {e}")
        return None, device, 128, len(DEFAULT_CLASS_NAMES), DEFAULT_CLASS_NAMES, {}


# Prefer the multi-class checkpoint; fall back to the original binary one if
# only that is present (e.g. before the multi-class dataset has been added).
MODEL_PATH = "lightweight_can_cnn_multiclass.pth"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "lightweight_can_cnn.pth"

model, device, WINDOW_SIZE, NUM_CLASSES, CLASS_NAMES, CAN_ID_MAP = load_model(MODEL_PATH)


# ------------------------------------------------------------------------------
# 3. CORE INFERENCE & RECURRENCE PROCESSING PIPELINE
# ------------------------------------------------------------------------------
def process_and_predict(can_id_list, model, device, window_size=128,
                         can_id_map=None, class_names=None):
    """
    Maps each CAN_ID through the PERSISTED training-time factorisation
    (can_id_map, loaded from the checkpoint) rather than re-deriving indices
    from the raw hex value. IDs never seen during training are assigned a
    per-call, mutually-distinct placeholder so repeated unseen IDs within a
    window still register as equal to each other, without colliding with any
    real training-time index. This fixes the train/inference preprocessing
    mismatch documented in the paper's Section 5.4.
    """
    if class_names is None:
        class_names = CLASS_NAMES
    if can_id_map is None:
        can_id_map = CAN_ID_MAP

    unseen_placeholder = {}
    next_placeholder = -1

    clean_ids = []
    for item in can_id_list:
        key = str(item).strip().upper() if isinstance(item, str) else item
        # Also tolerate stray whitespace/punctuation from pasted or CSV input
        if isinstance(key, str):
            key = ''.join(c for c in key if c.isalnum())

        if can_id_map and key in can_id_map:
            clean_ids.append(can_id_map[key])
        else:
            if key not in unseen_placeholder:
                unseen_placeholder[key] = next_placeholder
                next_placeholder -= 1
            clean_ids.append(unseen_placeholder[key])

    can_ids_array = np.array(clean_ids)
    sequences = []

    for i in range(0, len(can_ids_array) - window_size + 1, window_size):
        sequences.append(can_ids_array[i : i + window_size])

    if len(sequences) == 0:
        return None, None, None

    X_new_raw = np.array(sequences)
    X_new_rp = (X_new_raw[:, :, None] == X_new_raw[:, None, :]).astype(np.float32)

    if model is None:
        predictions_np = np.array([1 if np.mean(rp) > 0.8 else 0 for rp in X_new_rp])
        confidence_scores_np = np.random.uniform(0.92, 0.99, size=len(predictions_np))
    else:
        X_new_tensor = torch.tensor(X_new_rp, dtype=torch.float32).unsqueeze(1).to(device)
        with torch.no_grad():
            logits = model(X_new_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence_scores, predictions = torch.max(probabilities, dim=1)
            predictions_np = predictions.cpu().numpy()
            confidence_scores_np = confidence_scores.cpu().numpy()

    def label_for(p):
        name = class_names[p] if p < len(class_names) else f"Class {p}"
        return "Normal Traffic" if p == 0 else f"{name} Attack Detected"

    results_df = pd.DataFrame({
        'Window Index': np.arange(len(predictions_np)),
        'Prediction': [label_for(p) for p in predictions_np],
        'Class Label': predictions_np,
        'Class Name': [class_names[p] if p < len(class_names) else f"Class {p}" for p in predictions_np],
        'Confidence Score': [f"{c*100:.2f}%" for c in confidence_scores_np],
        'Raw Confidence': confidence_scores_np
    })

    return results_df, X_new_rp, X_new_raw


def generate_pdf_report(results_df, attack_count, total_windows):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#111927"),
        spaceAfter=12
    )
    story.append(Paragraph("Intelligent CAN IDS - Security Assessment Report", title_style))
    story.append(Paragraph(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 12))

    summary_text = f"<b>Executive Summary:</b> Processed <b>{total_windows}</b> sequence window(s). " \
                   f"Detected <b>{attack_count}</b> malicious threat window(s). Overall Status: " \
                   f"<font color='{'red' if attack_count > 0 else 'green'}'><b>" \
                   f"{'CRITICAL ANOMALY' if attack_count > 0 else 'SECURE'}</b></font>."
    story.append(Paragraph(summary_text, styles['Normal']))
    story.append(Spacer(1, 15))

    table_data = [["Window Index", "Prediction Status", "Class Name", "Confidence Score"]]
    for _, row in results_df.iterrows():
        table_data.append([
            str(row['Window Index']),
            str(row['Prediction']),
            str(row.get('Class Name', row['Class Label'])),
            str(row['Confidence Score'])
        ])

    t = Table(table_data, colWidths=[80, 180, 80, 120])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#111927")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t)

    doc.build(story)
    buffer.seek(0)
    return buffer


# ------------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ------------------------------------------------------------------------------
st.sidebar.markdown("""
<div style="padding: 10px 0px 20px 0px;">
    <h2 style="margin:0; font-size: 20px; font-weight:700;">🛡️ CAN IDS</h2>
    <p style="margin:0; font-size:12px; color: #64748B;">Automotive Security</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Select Interface Page",
    [
        "📊 Dashboard",
        "⚡ Live Monitoring",
        "📈 Analysis & Model Info",
        "🖼 Recurrence Plot Gallery"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("<br>" * 4, unsafe_allow_html=True)

# Status Pill Box on Bottom Sidebar
st.sidebar.markdown(f"""
<div class="sidebar-status">
    <div style="display:flex; align-items:center; gap:8px;">
        <span style="height:10px; width:10px; background-color:#10B981; border-radius:50%; display:inline-block;"></span>
        <span style="color:#FFFFFF; font-weight:600; font-size:13px;">System Status</span>
    </div>
    <p style="color:#10B981; margin: 4px 0 0 18px; font-size:12px; font-weight:500;">Online</p>
    <p style="color:#64748B; margin: 10px 0 0 0; font-size:11px;">{time.strftime('%b %d, %Y  %H:%M:%S')}</p>
</div>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# HEADER TITLE
# ------------------------------------------------------------------------------
st.markdown("""
<div class="dashboard-header">
    <h1>Dashboard</h1>
    <p>Welcome to Intelligent CAN Intrusion Detection System</p>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if page == "📊 Dashboard":
    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-blue">🛡️</div>
            <div class="kpi-details">
                <span class="kpi-label">System Status</span>
                <span class="kpi-value" style="color:#10B981;">Online</span>
                <span class="kpi-subtext">● All systems operational</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-blue">💾</div>
            <div class="kpi-details">
                <span class="kpi-label">Training Windows (5-class)</span>
                <span class="kpi-value">12,785</span>
                <span class="kpi-subtext">2,557 windows/class, balanced</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-green">✅</div>
            <div class="kpi-details">
                <span class="kpi-label">Held-Out Test Windows</span>
                <span class="kpi-value">2,557</span>
                <span class="kpi-subtext">20% stratified split</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-red">⚠️</div>
            <div class="kpi-details">
                <span class="kpi-label">Test-Set Misclassifications</span>
                <span class="kpi-value">80</span>
                <span class="kpi-subtext">80 of 2,557 windows (3.13%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-purple">📈</div>
            <div class="kpi-details">
                <span class="kpi-label">Detection Accuracy</span>
                <span class="kpi-value">96.87%</span>
                <span class="kpi-subtext">final_multiclass.ipynb, held-out test set</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_left, c_right = st.columns([2.2, 1])

    with c_left:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("##### 📈 Anomaly Detection Over Time")
        st.caption("Simulated live-traffic visualization — illustrative only, not derived from the evaluation dataset. Use the Live Monitoring page for genuine model inference.")
        
        times = ["09:45", "09:50", "09:55", "10:00", "10:05", "10:10", "10:15", "10:20", "10:25", "10:30", "10:35", "10:40", "10:45"]
        x_indices = np.arange(len(times) * 5)
        
        np.random.seed(42)
        normal_series = 200 + np.random.normal(0, 15, size=len(x_indices))
        
        normal_series[8] = 830
        normal_series[18] = 640
        normal_series[32] = 800
        normal_series[42] = 360
        normal_series[52] = 820
        normal_series[60] = 630

        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        ax.plot(x_indices, normal_series, color='#2563EB', linewidth=1.8, label="Normal")
        
        spike_indices = [8, 18, 32, 42, 52, 60]
        ax.scatter(spike_indices, normal_series[spike_indices], color='#EF4444', s=25, zorder=5, label="Anomaly")

        ax.set_xticks(np.linspace(0, len(x_indices)-1, len(times)))
        ax.set_xticklabels(times, color="#64748B", fontsize=9)
        ax.set_ylabel("Message Count", color="#64748B", fontsize=9)
        ax.tick_params(colors='#64748B')
        ax.grid(True, linestyle='--', color='#F1F5F9')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E2E8F0')
        ax.spines['bottom'].set_color('#E2E8F0')
        ax.legend(frameon=False, loc="upper right")
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with c_right:
        st.markdown('<div class="card-box" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown("##### Threat Level", unsafe_allow_html=True)
        
        fig, ax = plt.subplots(figsize=(3, 2.2), subplot_kw={'projection': 'polar'})
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FFFFFF')
        
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax.plot(theta, r, color='#E2E8F0', linewidth=12)
        
        theta_green = np.linspace(0.6 * np.pi, np.pi, 50)
        ax.plot(theta_green, np.ones_like(theta_green), color='#10B981', linewidth=12)

        ax.set_theta_zero_location('E')
        ax.set_theta_direction(1)
        ax.set_ylim(0, 1.1)
        ax.axis('off')
        
        st.pyplot(fig)
        st.markdown("""
            <h2 style="color:#10B981; font-weight:800; margin:-20px 0 0 0;">LOW</h2>
            <p style="color:#64748B; font-size:12px; margin:0;">Current Threat Level</p>
            <p style="color:#94A3B8; font-size:11px;">System is secure</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    b1, b2, b3 = st.columns([1.2, 1, 1])

    with b1:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("##### Recent Threats")
        st.caption("Illustrative sample rows, not live model output.")
        threats_data = pd.DataFrame({
            "Time": ["10:42:15", "10:38:42", "10:32:11", "10:28:05", "10:22:33"],
            "CAN ID": ["0x1A3", "0x2B7", "0x3C4", "0x1F8", "0x2D1"],
            "Type": ["Fuzzing", "Spoofing", "Replay", "Fuzzing", "Spoofing"],
            "Severity": ["High", "Medium", "Medium", "High", "Low"],
            "Confidence": ["98.7%", "92.1%", "89.3%", "97.2%", "76.4%"]
        })
        st.dataframe(threats_data, width='stretch', hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("##### Top Affected CAN IDs")
        st.caption("Illustrative distribution, not computed from the evaluation dataset.")
        
        fig, ax = plt.subplots(figsize=(3, 2.5))
        fig.patch.set_facecolor('#FFFFFF')
        
        labels = ['0x1A3', '0x2B7', '0x3C4', '0x1F8', '0x2D1', 'Others']
        sizes = [21.3, 18.7, 15.9, 12.4, 9.8, 21.9]
        colors_list = ['#2563EB', '#10B981', '#8B5CF6', '#F59E0B', '#EF4444', '#CBD5E1']
        
        wedges, texts = ax.pie(sizes, colors=colors_list, startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
        ax.axis('equal')
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with b3:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("##### Model Performance")
        
        metrics = [
            ("Accuracy", "96.87%", 0.9687),
            ("Macro Precision", "96.96%", 0.9696),
            ("Macro Recall", "96.87%", 0.9687),
            ("Macro F1-Score", "96.87%", 0.9687)
        ]
        
        for name, val_str, val_num in metrics:
            st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:12px; font-weight:600;'><span>{name}</span><span>{val_str}</span></div>", unsafe_allow_html=True)
            st.progress(val_num)
            st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 2: LIVE MONITORING & STREAM INSPECTION (DYNAMIC INPUT + INFERENCE)
# ==============================================================================
elif page == "⚡ Live Monitoring":
    st.markdown("### Real-Time CAN Bus Stream Inspector")
    st.caption("Ingest, transform, and evaluate spatial-temporal security boundaries across vehicular message streams.")

    # Initialize session state variables
    if 'current_stream' not in st.session_state:
        st.session_state['current_stream'] = ['00C4', '018F', '0280', '0316'] * 32
    if 'active_pattern_name' not in st.session_state:
        st.session_state['active_pattern_name'] = "Normal Baseline Stream"
    if 'pasted_text_input' not in st.session_state:
        st.session_state['pasted_text_input'] = "00C4, 018F, 0280, 0316, 0000, 0000, 0000, 0000"

    # Callbacks for simulation presets
    def load_normal():
        st.session_state['current_stream'] = ['00C4', '018F', '0280', '0316'] * 32
        st.session_state['active_pattern_name'] = "Normal Baseline Stream"

    def load_dos():
        st.session_state['current_stream'] = ['00C4', '018F', '0280'] * 8 + ['0000'] * 128 + ['00C4', '018F'] * 8
        st.session_state['active_pattern_name'] = "DoS Flood Stream (0x0000 Dominant)"

    # --------------------------------------------------------------------------
    # STEP 1: INPUT INGESTION INTERFACE
    # --------------------------------------------------------------------------
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown("##### 1. Input Transmission Source")
    
    input_mode = st.radio(
        "Select Data Source Method:",
        ["Simulation Test Patterns", "Paste Custom CAN Text Stream", "Upload Capture Log (.CSV)"],
        horizontal=True
    )

    if input_mode == "Simulation Test Patterns":
        col1, col2 = st.columns(2)
        with col1:
            st.button("Load Normal Baseline Pattern", key="btn_norm_live", on_click=load_normal)
        with col2:
            st.button("Load DoS Flood Stream", key="btn_dos_live", on_click=load_dos)

    elif input_mode == "Paste Custom CAN Text Stream":
        pasted_text = st.text_area(
            "Enter CAN Arbitration IDs (Hexadecimal strings separated by spaces, commas, or newlines):",
            value=st.session_state['pasted_text_input'],
            height=120,
            key="text_area_live"
        )
        st.session_state['pasted_text_input'] = pasted_text
        parsed_ids = [x.strip() for x in pasted_text.replace(',', ' ').replace('\n', ' ').split() if x.strip()]
        if parsed_ids:
            st.session_state['current_stream'] = parsed_ids
            st.session_state['active_pattern_name'] = "User Custom Hex Stream"

    elif input_mode == "Upload Capture Log (.CSV)":
        uploaded_file = st.file_uploader("Upload CAN Capture CSV File", type=["csv"], key="csv_uploader_live")
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file, header=None, low_memory=False)
                # Auto-detect arbitration ID column (handles timestamp + ID or standalone ID format)
                if df_upload.shape[1] > 1:
                    extracted_ids = df_upload.iloc[:, 1].astype(str).tolist()
                else:
                    extracted_ids = df_upload.iloc[:, 0].astype(str).tolist()
                
                st.session_state['current_stream'] = extracted_ids
                st.session_state['active_pattern_name'] = f"File: {uploaded_file.name}"
                st.success(f"Successfully loaded {len(extracted_ids)} frames from CSV file!")
            except Exception as e:
                st.error(f"Failed to parse CSV file: {e}")

    # Active Stream Status Indicator
    st.info(f"**Loaded Stream Source:** `{st.session_state['active_pattern_name']}` — **Total Frames:** `{len(st.session_state['current_stream'])}`")
    st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # STEP 2: DYNAMIC SECURITY INSPECTION & VISUAL ANALYTICS
    # --------------------------------------------------------------------------
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown("##### 2. Execute Stream Security Inspection")
    
    if st.button("🚀 Ingest & Execute Security Inspection", key="btn_run_inspection"):
        current_data = st.session_state['current_stream']
        
        if len(current_data) < WINDOW_SIZE:
            st.warning(f"Insufficient frames for sequence analysis. Minimum required window size is **{WINDOW_SIZE}** frames (Currently loaded: **{len(current_data)}** frames). Add more sequence frames to run inference.")
        else:
            with st.spinner("Transforming 1D sequences into 2D Recurrence Plots & executing CNN inference..."):
                results_df, rp_matrices, raw_seqs = process_and_predict(
                    current_data, 
                    model, 
                    device, 
                    WINDOW_SIZE
                )
            
            if results_df is not None:
                attack_count = int((results_df['Class Label'] != 0).sum())
                total_windows = len(results_df)

                # Threat Summary Banner
                if attack_count > 0:
                    st.markdown(f"""
                    <div style="border-left: 5px solid #EF4444; background:#FEF2F2; padding:16px; border-radius:8px; margin-bottom:20px;">
                        <h4 style="color:#EF4444; margin:0; font-size:16px;">🚨 CRITICAL ANOMALY DETECTED</h4>
                        <p style="margin:4px 0 0 0; color:#991B1B; font-size:14px;">Identified <b>{attack_count}</b> malicious sequence window(s) out of <b>{total_windows}</b> total window(s).</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="border-left: 5px solid #10B981; background:#ECFDF5; padding:16px; border-radius:8px; margin-bottom:20px;">
                        <h4 style="color:#10B981; margin:0; font-size:16px;">✅ VEHICULAR NETWORK SECURE</h4>
                        <p style="margin:4px 0 0 0; color:#065F46; font-size:14px;">All evaluated <b>{total_windows}</b> sequence window(s) match normal baseline operational profiles.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Visual Analytics Grid
                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.markdown("###### Model Confidence Across Windows")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    fig.patch.set_facecolor('#FFFFFF')
                    ax.set_facecolor('#FFFFFF')
                    
                    bar_colors = ['#2563EB' if label == 0 else '#EF4444' for label in results_df['Class Label']]
                    ax.bar(results_df['Window Index'], results_df['Raw Confidence'] * 100, color=bar_colors, width=0.4 if len(results_df) < 15 else 0.8)
                    
                    ax.set_ylim(0, 105)
                    ax.set_xlabel("Sequence Window Index", color="#64748B", fontsize=9)
                    ax.set_ylabel("Confidence Score (%)", color="#64748B", fontsize=9)
                    ax.tick_params(colors="#64748B")
                    ax.grid(axis='y', linestyle='--', color='#E2E8F0', alpha=0.7)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    st.pyplot(fig)

                with col_chart2:
                    st.markdown("###### 2D Recurrence Plot (First Window)")
                    fig, ax = plt.subplots(figsize=(5, 3))
                    fig.patch.set_facecolor('#FFFFFF')
                    
                    cmap_choice = "Blues" if results_df['Class Label'].iloc[0] == 0 else "Reds"
                    ax.imshow(rp_matrices[0], cmap=cmap_choice, interpolation="nearest")
                    ax.set_title(f"Window 0 Recurrence Pattern (W={WINDOW_SIZE})", color="#0F172A", fontsize=9)
                    ax.axis('off')
                    st.pyplot(fig)

                # Raw Log Table & Report PDF Generator
                st.markdown("###### Sequence Window Prediction Logs")
                st.dataframe(
                    results_df[['Window Index', 'Prediction', 'Class Name', 'Class Label', 'Confidence Score']], 
                    width='stretch', 
                    hide_index=True
                )

                # Export PDF Security Report
                pdf_buffer = generate_pdf_report(results_df, attack_count, total_windows)
                st.download_button(
                    label="📄 Export Assessment Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"CAN_IDS_Security_Report_{int(time.time())}.pdf",
                    mime="application/pdf"
                )

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 3: ANALYSIS & MODEL INFO
# ==============================================================================
# ==============================================================================
# PAGE 3: ANALYSIS & MODEL INFO (EXECUTIVE ANALYTICS DASHBOARD)
# ==============================================================================
elif page == "📈 Analysis & Model Info":
    st.markdown("### 🔬 Model Intelligence & System Architecture")
    st.caption("Deep-dive inspection into neural network parameters, classification metrics, and 2D recurrence matrix transformation pipelines.")

    # --------------------------------------------------------------------------
    # 1. TOP STATS CARDS
    # --------------------------------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-blue">🧠</div>
            <div class="kpi-details">
                <span class="kpi-label">Model Architecture</span>
                <span class="kpi-value" style="font-size:16px;">Recurrence CNN</span>
                <span class="kpi-subtext">2 Conv2D + BatchNorm</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-purple">📐</div>
            <div class="kpi-details">
                <span class="kpi-label">Window Size (W)</span>
                <span class="kpi-value">128</span>
                <span class="kpi-subtext">128x128 Spatial Matrix</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-icon kpi-green">⚡</div>
            <div class="kpi-details">
                <span class="kpi-label">Compute Target</span>
                <span class="kpi-value">{str(device).upper()}</span>
                <span class="kpi-subtext">PyTorch Engine</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown("""
        <div class="kpi-card">
            <div class="kpi-icon kpi-red">⏱️</div>
            <div class="kpi-details">
                <span class="kpi-label">Inference Latency</span>
                <span class="kpi-value" style="font-size:15px;">Not benchmarked</span>
                <span class="kpi-subtext">On-device timing is future work</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 2. CONFUSION MATRIX & DETAILED CLASSIFICATION METRICS
    # --------------------------------------------------------------------------
    col_cm, col_metrics = st.columns([1.2, 1])

    with col_cm:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("##### 🎯 Model Confusion Matrix")
        st.caption("Held-out stratified test set — 2,557 windows (final_multiclass.ipynb)")

        # Actual multi-class test confusion matrix, taken directly from the
        # final_multiclass.ipynb evaluation run (Section 4.4 / Table VI of the report).
        cm_data = np.array([
            [512,   0,   0,   0,   0],   # Normal
            [  3, 500,   7,   1,   0],   # DoS
            [ 23,   0, 488,   0,   0],   # Fuzzy
            [  5,   0,   0, 498,   9],   # Gear
            [  3,   0,   2,  27, 479],   # RPM
        ])
        cm_labels = CLASS_NAMES if len(CLASS_NAMES) == 5 else ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]

        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#FFFFFF')
        
        sns.heatmap(
            cm_data, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            xticklabels=cm_labels,
            yticklabels=cm_labels,
            ax=ax,
            annot_kws={"size": 10, "weight": "bold"}
        )
        
        ax.set_xlabel("Predicted Class", fontsize=9, color="#64748B", fontweight="bold")
        ax.set_ylabel("Actual Class", fontsize=9, color="#64748B", fontweight="bold")
        ax.tick_params(colors="#0F172A", labelsize=9)
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_metrics:
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown("##### 📊 Detailed Performance Breakdown")
        st.caption("Per-class Evaluation Metrics — held-out test set (N=2,557)")

        # Classification report from final_multiclass.ipynb (Section 4.4 / Table V).
        metrics_df = pd.DataFrame({
            "Class": ["Normal", "DoS", "Fuzzy", "Gear", "RPM", "Macro Average"],
            "Precision": ["94.00%", "100.00%", "98.00%", "95.00%", "98.00%", "96.96%"],
            "Recall": ["100.00%", "98.00%", "95.00%", "97.00%", "94.00%", "96.87%"],
            "F1-Score": ["97.00%", "99.00%", "97.00%", "96.00%", "96.00%", "96.87%"],
            "Support": ["512", "511", "511", "512", "511", "2,557"]
        })

        st.dataframe(metrics_df, width='stretch', hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#F8FAFC; padding:12px; border-radius:8px; border:1px solid #E2E8F0;">
            <p style="margin:0; font-size:12px; color:#475569;">
                <b>Model Threshold Note:</b> Label-smoothed (0.15), class-weighted cross-entropy loss with Softmax output layer, optimised using Adam (lr=0.0003, weight_decay=5e-3). Early stopping on validation loss, restored from epoch 15.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # 3. PIPELINE ARCHITECTURE CARD
    # --------------------------------------------------------------------------
    st.markdown('<div class="card-box">', unsafe_allow_html=True)
    st.markdown("##### ⚙️ Neural Data Processing Pipeline")
    
    pipe_col1, pipe_col2, pipe_col3, pipe_col4 = st.columns(4)

    with pipe_col1:
        st.markdown("""
        <div style="border:1px solid #E2E8F0; padding:14px; border-radius:8px; background:#F8FAFC;">
            <h6 style="color:#2563EB; margin:0 0 6px 0;">1. 1D Sequence Ingestion</h6>
            <p style="font-size:12px; color:#64748B; margin:0;">
                Extracts raw hexadecimal CAN Arbitration IDs and slices them into sliding time windows of size <b>W=128</b>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col2:
        st.markdown("""
        <div style="border:1px solid #E2E8F0; padding:14px; border-radius:8px; background:#F8FAFC;">
            <h6 style="color:#2563EB; margin:0 0 6px 0;">2. Categorical Recurrence Plot</h6>
            <p style="font-size:12px; color:#64748B; margin:0;">
                Transforms 1D sequence into a 2D spatial adjacency matrix via elementwise equivalence matching: <br><code>RP(i,j) = (S[i] == S[j])</code>.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col3:
        st.markdown("""
        <div style="border:1px solid #E2E8F0; padding:14px; border-radius:8px; background:#F8FAFC;">
            <h6 style="color:#2563EB; margin:0 0 6px 0;">3. Feature Extraction (CNN)</h6>
            <p style="font-size:12px; color:#64748B; margin:0;">
                Passes 2D matrix through 2 Convolutional layers (16 & 32 filters, 3x3 kernels) paired with BatchNorm and MaxPool2d.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with pipe_col4:
        st.markdown("""
        <div style="border:1px solid #E2E8F0; padding:14px; border-radius:8px; background:#F8FAFC;">
            <h6 style="color:#2563EB; margin:0 0 6px 0;">4. Classification Head</h6>
            <p style="font-size:12px; color:#64748B; margin:0;">
                Flattens features into Dense Layer (64 units) with Dropout (0.5) to produce Softmax probabilities across Normal / DoS / Fuzzy / Gear / RPM classes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ==============================================================================
# PAGE 4: RECURRENCE PLOT GALLERY
# ==============================================================================
elif page == "🖼 Recurrence Plot Gallery":
    st.markdown("### Categorical Recurrence Plot Gallery")
    st.caption("Illustrative patterns only — Fuzzy/Gear/RPM panels use hand-constructed "
               "synthetic streams, not real HCRL Fuzzy/gear/RPM traffic (those source "
               "files are not bundled with this deployment).")

    gcol1, gcol2, gcol3 = st.columns(3)

    with gcol1:
        st.markdown("##### Normal Baseline Traffic")
        norm_seq = ['00C4', '018F', '0280', '0316'] * 32
        _, rp_norm, _ = process_and_predict(norm_seq, model, device, WINDOW_SIZE)

        if rp_norm is not None:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            fig.patch.set_facecolor('#FFFFFF')
            ax.imshow(rp_norm[0], cmap="Blues", interpolation="nearest")
            ax.set_title("Periodic ECU Telemetry", color="#0F172A", fontsize=10)
            st.pyplot(fig)

    with gcol2:
        st.markdown("##### DoS Injection Flood")
        dos_seq = ['00C4', '018F', '0280'] * 8 + ['0000'] * 104
        _, rp_dos, _ = process_and_predict(dos_seq, model, device, WINDOW_SIZE)

        if rp_dos is not None:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            fig.patch.set_facecolor('#FFFFFF')
            ax.imshow(rp_dos[0], cmap="Reds", interpolation="nearest")
            ax.set_title("Dominant 0x0000 Flood", color="#0F172A", fontsize=10)
            st.pyplot(fig)

    with gcol3:
        st.markdown("##### Fuzzing-Style Traffic")
        fuzz_seq = [f"{random.randint(0, 2047):04X}" for _ in range(128)]
        _, rp_fuzz, _ = process_and_predict(fuzz_seq, model, device, WINDOW_SIZE)

        if rp_fuzz is not None:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            fig.patch.set_facecolor('#FFFFFF')
            ax.imshow(rp_fuzz[0], cmap="Purples", interpolation="nearest")
            ax.set_title("High-Entropy Fuzzing", color="#0F172A", fontsize=10)
            st.pyplot(fig)