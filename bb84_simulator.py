import io
import textwrap
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Optional

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator, Aer
    from qiskit.visualization import plot_bloch_multivector
    QISKIT_OK = True
except Exception:
    QISKIT_OK = False


st.set_page_config(
    page_title="BB84 Quantum Key Distribution Simulator",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://quantum-computing.ibm.com/',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': """
        # BB84 Quantum Key Distribution Simulator
        
        An interactive educational tool demonstrating quantum cryptography principles 
        using the BB84 protocol. Developed for the Amaravati Quantum Valley Hackathon 2025.
        
        **Version:** 1.0.0  
        **License:** MIT  
        """
    }
)



# Minimal custom CSS to elevate the look
st.markdown(
    """
<style>
/* Base styling */
.main > div {
padding-top: 1.5rem;
max-width: 1200px;
margin: 0 auto;
}

/* Gradient background with subtle animation */
.stApp {
    background: radial-gradient(1200px 800px at 10% 20%, rgba(238, 242, 255, 0.8) 0%, transparent 60%),
    radial-gradient(1400px 900px at 110% 10%, rgba(245, 243, 255, 0.8) 0%, transparent 60%);
    background-attachment: fixed;
    animation: gradientShift 20s ease infinite alternate;
}

@keyframes gradientShift {
    0% { background-position: 0% 0%, 100% 100%; }
    100% { background-position: 10% 10%, 90% 90%; }
}

/* Header styling */
.app-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 1.5rem;
}

/* Modern card design with hover effect */
.card {
    background: rgba(255, 255, 255, 0.9);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 4px 30px rgba(2, 6, 23, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(2, 6, 23, 0.15);
}

/* Pill/tag design */
.pill {
    background: linear-gradient(135deg, #e2e8f0 50%, #f1f5f9 100%);
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 500;
    color: #334155;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* Footer styling */
.footer {
    color: #475569;
    text-align: center;
    font-size: 13px;
    margin-top: 48px;
    padding: 24px 0;
    border-top: 1px solid rgba(226, 232, 240, 0.5);
}

/* Metric display */
.metric {
    font-weight: 700;
    font-size: 32px;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 8px 0;
}

/* Muted text */
.muted {
    color: #64748b;
    font-size: 14px;
}

/* Code chip with animation */
.code-chip {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    color: white;
    padding: 4px 12px;
    border-radius: 8px;
    font-size: 13px;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    box-shadow: 0 2px 4px rgba(14, 165, 233, 0.2);
    transition: all 0.2s ease;
}

.code-chip:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 8px rgba(14, 165, 233, 0.3);
}

/* Input field styling */
.stTextInput>div>div>input, 
.stNumberInput>div>div>input,
.stSelectbox>div>div>select {
    border-radius: 12px !important;
    padding: 10px 14px !important;
    border: 1px solid #e2e8f0 !important;
}

/* Button styling */
.stButton>button {
    border-radius: 12px !important;
    padding: 10px 24px !important;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
    transition: all 0.2s ease !important;
}

.stButton>button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3) !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(241, 245, 249, 0.5);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}
</style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Helpers & cache
# ---------------------------
@dataclass
class RunResult:
    alice_bits: List[int]
    alice_bases: List[str]
    bob_bits: List[int]
    bob_bases: List[str]
    sifted_alice: List[int]
    sifted_bob: List[int]
    final_key: List[int]
    error_rate: float
    sample_indices: List[int]


def _random_bits(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.int8)


def _random_bases(n: int, rng: np.random.Generator) -> np.ndarray:
    # '+' or '×'
    return np.where(rng.random(n) < 0.5, '+', '×')


@st.cache_resource(show_spinner=False)
def get_simulators():
    if not QISKIT_OK:
        return None, None
    qasm_sim = AerSimulator()
    sv_sim = Aer.get_backend('statevector_simulator')
    return qasm_sim, sv_sim


# ----------------------------------
# Core BB84 simulation (efficient)
# ----------------------------------



def run_bb84(num_bits: int, eve_present: bool, seed: Optional[int] = None) -> RunResult:

    rng = np.random.default_rng(seed)

    # Step 1: Alice bits & bases
    alice_bits = _random_bits(num_bits, rng).tolist()
    alice_bases = _random_bases(num_bits, rng).tolist()

    # Prepare qubits as circuits once
    qubits: List[QuantumCircuit] = []
    if QISKIT_OK:
        for bit, basis in zip(alice_bits, alice_bases):
            qc = QuantumCircuit(1)
            if bit == 1:
                qc.x(0)
            if basis == '×':
                qc.h(0)
            qubits.append(qc)
    else:
        qubits = []  # Placeholder if qiskit missing

    # Step 3: Quantum channel
    bob_bases = _random_bases(num_bits, rng).tolist()
    bob_bits: List[int] = []
    eve_bits: List[int] = []

    if QISKIT_OK:
        qasm_sim, _ = get_simulators()
        for i in range(num_bits):
            qc = qubits[i].copy()

            # Eve
            if eve_present:
                eve_basis = '+' if rng.random() < 0.5 else '×'
                if eve_basis == '×':
                    qc.h(0)
                qc.measure_all()
                result = qasm_sim.run(transpile(qc, qasm_sim), shots=1).result()
                eve_bit = int(list(result.get_counts())[0])
                eve_bits.append(eve_bit)
                # Re-prepare
                qc = QuantumCircuit(1)
                if eve_bit == 1:
                    qc.x(0)
                if eve_basis == '×':
                    qc.h(0)

            # Bob measurement
            if bob_bases[i] == '×':
                qc.h(0)
            qc.measure_all()
            result = qasm_sim.run(transpile(qc, qasm_sim), shots=1).result()
            bob_bit = int(list(result.get_counts())[0])
            bob_bits.append(bob_bit)
    else:
        # Fallback: classical approximation if Qiskit is unavailable
        for i in range(num_bits):
            # If bases match, Bob gets Alice's bit; if not, random
            if alice_bases[i] == bob_bases[i]:
                bob_bits.append(alice_bits[i])
            else:
                bob_bits.append(int(np.random.random() < 0.5))

    # Step 4: Sifting
    sifted_alice_bits: List[int] = []
    sifted_bob_bits: List[int] = []
    for i in range(num_bits):
        if alice_bases[i] == bob_bases[i]:
            sifted_alice_bits.append(alice_bits[i])
            sifted_bob_bits.append(bob_bits[i])

    # Step 5: Error estimation (sample up to 20% or 10, whichever smaller)
    if len(sifted_alice_bits) > 0:
        sample_size = min(max(1, len(sifted_alice_bits) // 5), 10)
        rng2 = np.random.default_rng(seed + 1 if seed is not None else None)
        sample_indices = rng2.choice(len(sifted_alice_bits), size=sample_size, replace=False).tolist()
        error_count = sum(1 for idx in sample_indices if sifted_alice_bits[idx] != sifted_bob_bits[idx])
        error_rate = error_count / sample_size if sample_size > 0 else 0.0
        final_key = [bit for i, bit in enumerate(sifted_alice_bits) if i not in sample_indices]
    else:
        sample_indices = []
        error_rate = 0.0
        final_key = []

    return RunResult(
        alice_bits, alice_bases, bob_bits, bob_bases,
        sifted_alice_bits, sifted_bob_bits, final_key, error_rate, sample_indices
    )


# ----------------------------------
# UI Sections
# ----------------------------------

def header():
    st.markdown(
        """
<style>
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-5px); }
    100% { transform: translateY(0px); }
}
.quantum-icon {
    animation: float 3s ease-in-out infinite;
    filter: drop-shadow(0 4px 6px rgba(79, 70, 229, 0.2));
}
.protocol-tag {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    padding: 4px 16px;
    border-radius: 999px;
    font-size: 14px;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3);
    transition: all 0.3s ease;
}
.protocol-tag:hover {
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
}
.header-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
}
.header-subtitle {
    color: #64748b;
    font-size: 1.1rem;
    margin-top: 8px;
    line-height: 1.5;
    max-width: 800px;
}
.header-container {
    padding-bottom: 1.5rem;
    border-bottom: 1px solid rgba(226, 232, 240, 0.5);
    margin-bottom: 2rem;
}
</style>

<div class="header-container">
<div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
<span class="protocol-tag">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM12 20C7.59 20 4 16.41 4 12C4 7.59 7.59 4 12 4C16.41 4 20 7.59 20 12C20 16.41 16.41 20 12 20Z" fill="white"/>
<path d="M12 6C8.69 6 6 8.69 6 12C6 15.31 8.69 18 12 18C15.31 18 18 15.31 18 12C18 8.69 15.31 6 12 6ZM12 16C9.79 16 8 14.21 8 12C8 9.79 9.79 8 12 8C14.21 8 16 9.79 16 12C16 14.21 14.21 16 12 16Z" fill="white"/>
</svg>
BB84 PROTOCOL
</span>
<h1 class="header-title">Quantum Key Distribution Simulator</h1>
<span class="quantum-icon">🔮</span>
</div>
<p class="header-subtitle">
Interactively explore the BB84 protocol with quantum state visualization, eavesdropping detection, 
and secure key generation. See quantum cryptography in action through animated Bloch spheres 
and real-time protocol simulation.
</p>
</div>
        """,
        unsafe_allow_html=True
    )




def sidebar_nav() -> str:
    # --- Custom CSS for sidebar ---
    st.markdown(
        """
<style>
/* Sidebar container */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0e1f 0%, #5935b5 100%);
    padding: 1rem;
}

/* Logo styling */
.sidebar-logo {
    filter: drop-shadow(0 4px 6px rgba(79, 70, 229, 0.2));
    transition: all 0.3s ease;
    margin-bottom: 1.5rem;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(226, 232, 240, 0.5);
}
/* Navigation title */
.sidebar-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #3b32d1;
    margin: 1rem 0;
    text-align: center;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 0.5rem;
}
/* Radio container */
div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 8px !important;
}
/* Radio button label */
div[role="radiogroup"] > label {
    padding: 12px 18px !important;
    border-radius: 10px !important;
    cursor: pointer;
    font-weight: 500 !important;
    color: #10448f !important;
    border: 1px solid transparent !important;
    transition: all 0.3s ease !important;
    margin: 0 !important;
}
/* Hover effect */
div[role="radiogroup"] > label:hover {
    background: rgba(238, 242, 255, 0.9) !important;
    transform: translateX(4px) !important;
}
/* Selected option */
div[role="radiogroup"] > label[data-baseweb="radio"]:has(> div:first-child[aria-checked="true"]) {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    box-shadow: 0 4px 10px rgba(99, 102, 241, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}
/* Footer */
.sidebar-footer {
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #062654;
    text-align: center;
    padding-top: 1rem;
    border-top: 1px solid #e2e8f0;
}
</style>
        """,
        unsafe_allow_html=True
    )

    # --- Sidebar Layout ---
    with st.sidebar:
        # Logo
        st.markdown(
            """
<div style="text-align: center">
<img class="sidebar-logo" 
src="https://upload.wikimedia.org/wikipedia/commons/6/6b/Bloch_sphere.svg" 
width="100%">
</div>
<div class="sidebar-title">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M3 9L12 2L21 9V20C21 20.5304 20.7893 21.0391 20.4142 21.4142C20.0391 21.7893 19.5304 22 19 22H5C4.46957 22 3.96086 21.7893 3.58579 21.4142C3.21071 21.0391 3 20.5304 3 20V9Z" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M9 22V12H15V22" stroke="#4f46e5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
NAVIGATION
</div>
            """,
            unsafe_allow_html=True
        )

        # Initialize session state for page navigation
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 'Home'

        # Navigation options with emoji icons
        nav_options = {
            "🏠 Home": "Home",
            "⚛️ Simulator": "Simulator", 
            "📚 Theory": "Theory",
            "👥 Team": "Team",
            "❓ FAQ": "FAQ",
            "✉️ Contact": "Contact"
        }

        # Create radio buttons for navigation
        selected = st.radio(
            "Navigate to:",
            options=list(nav_options.keys()),
            label_visibility="collapsed",
            key="nav_radio"
        )

        # Update current page in session state
        st.session_state.current_page = nav_options[selected]

        # Footer
        st.markdown(
"""<div class="sidebar-footer">
Quantum Security Lab © 2025<br>
<span style="font-size:0.7rem">v1.0.0</span>
</div>""",
            unsafe_allow_html=True
        )

    return st.session_state.current_page

def section_home():
    st.markdown(
        """
    <style>
.feature-card {
    background: linear-gradient(135deg, rgba(20, 30, 48, 0.95) 0%, rgba(36, 59, 85, 0.95) 100%);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
    height: 100%;
    transition: all 0.3s ease;
    backdrop-filter: blur(8px);
}
.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(100, 108, 255, 0.3);
}
.feature-card h3 {
    color: white;
    margin-top: 0;
    margin-bottom: 16px;
    font-size: 1.4rem;
    display: flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(90deg, #646cff 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.feature-card ul, .feature-card ol {
    padding-left: 24px;
    margin-bottom: 16px;
}
.feature-card li {
    margin-bottom: 8px;
    color: #e2e8f0;
    position: relative;
}
.feature-card li::before {
    content: "•";
    color: #818cf8;
    font-weight: bold;
    display: inline-block;
    width: 1em;
    margin-left: -1em;
}
.feature-card .muted {
    font-size: 0.9rem;
    color: #94a3b8;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px dashed rgba(148, 163, 184, 0.3);
}
.emoji-icon {
    font-size: 1.5rem;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.2));
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.quick-tip {
    background: rgba(30, 41, 59, 0.7);
    border-left: 4px solid #818cf8;
    padding: 12px;
    border-radius: 0 8px 8px 0;
    margin-top: 16px;
    color: #e2e8f0;
    font-size: 0.95rem;
}
.quick-tip strong {
    color: white;
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
    </style>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.15, 1], gap="large")
    
    with col1:
        st.markdown(
            """
<div class="feature-card">
<h3><span class="emoji-icon">👋</span> Welcome to Quantum Secure</h3>
<p style="color: #475569; margin-bottom: 20px;">This interactive simulator demonstrates the BB84 quantum key distribution protocol—the gold standard for quantum-safe cryptography. Explore how quantum principles enable theoretically unbreakable encryption.</p>
<ul>
<li><b>Generate quantum states</b> using superposition and basis choices</li>
<li><b>Detect eavesdroppers</b> through quantum measurement disturbance</li>
<li><b>Visualize qubits</b> on animated Bloch spheres</li>
<li><b>Extract secure keys</b> through sifting and error estimation</li>
</ul>
<div class="quick-tip">
<b>Pro Tip:</b> Compare error rates with and without Eve to see quantum security in action!
</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
<div class="feature-card">
<h3><span class="emoji-icon">⚡</span> Quick Start Guide</h3>
<ol>
<li><b>Navigate to Simulator</b> using the sidebar menu</li>
<li><b>Configure parameters</b>:
<ul style="padding-left: 20px; margin-top: 8px;">
<li>Set key length (20-100 qubits)</li>
<li>Toggle Eve to simulate attacks</li>
<li>Enable Bloch sphere visualization</li>
</ul>
</li>
<li><b>Run protocol</b> and analyze results:
<ul style="padding-left: 20px; margin-top: 8px;">
<li>View sifted key statistics</li>
<li>Check eavesdropper detection</li>
<li>Download final secure key</li>
</ul>
</li>
</ol>
<div class="muted">
<svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="vertical-align: middle; margin-right: 6px;">
<path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M12 8V12" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M12 16H12.01" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
Tip: For best experience, view on desktop with Chrome or Edge
</div>
</div>
            """,
            unsafe_allow_html=True,
        )


def plot_bloch_samples(alice_bits: List[int], alice_bases: List[str]):
    if not QISKIT_OK:
        st.warning(
            """
            **Quantum visualization requires Qiskit**  
            Install with:  
            ```bash
            pip install qiskit qiskit-aer qiskit-visualization matplotlib
            ```
            """
        )
        return

    try:
        _, sv_sim = get_simulators()
        cols = st.columns(3, gap="medium")
        
        for i in range(min(3, len(alice_bits))):
            with cols[i]:
                # Create quantum circuit
                qc = QuantumCircuit(1)
                if alice_bits[i] == 1:
                    qc.x(0)
                if alice_bases[i] == '×':
                    qc.h(0)
                
                # Get statevector
                result = sv_sim.run(transpile(qc, sv_sim)).result()
                statevector = result.get_statevector()
                
                # Create Bloch sphere visualization
                fig = plt.figure(figsize=(4, 4), facecolor='none')
                ax = fig.add_subplot(111, projection='3d')
                
                # Customize Bloch sphere appearance
                plot_bloch_multivector(statevector)
                plt.show()
                ax.set_title(f"Qubit {i+1}", y=1.08, fontsize=14, color='#0f172a')
                
                # Style adjustments
                ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
                ax.grid(False)
                
                # Add info box
                info_text = f"""
<div style="
background: rgba(248, 250, 252, 0.8);
border-radius: 8px;
padding: 8px;
margin-top: 8px;
font-size: 12px;
color: #334155;
">
<b>State:</b> |{alice_bits[i]}⟩<br>
<b>Basis:</b> {alice_bases[i]} ({'Hadamard' if alice_bases[i] == '×' else 'Standard'})<br>
<b>Vector:</b> {np.round(statevector, 3)}
</div>
                """
                st.markdown(info_text, unsafe_allow_html=True)
                
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
    except Exception as e:
        st.error(f"Error rendering Bloch spheres: {str(e)}")
        st.code(f"Debug info:\n{alice_bits[:3]}\n{alice_bases[:3]}", language='python')


def section_simulator():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Controls
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        num_bits = st.slider("Number of raw bits", 10, 500, 64, step=2)
    with c2:
        eve_present = st.toggle("Enable Eve", value=False)
    with c3:
        show_bloch = st.toggle("Show Bloch spheres", value=False)
    with c4:
        seed = st.number_input("Random seed (optional)", value=0, step=1)
        seed = int(seed) if seed != 0 else None

    run = st.button("Run BB84", type="primary")

    st.markdown("</div>", unsafe_allow_html=True)

    if run:
        with st.spinner("Simulating BB84…"):
            result = run_bb84(num_bits=num_bits, eve_present=eve_present, seed=seed)

        # Metrics
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Raw length", f"{num_bits}")
            st.caption("Alice's initial bits")
        with m2:
            st.metric("Sifted length", f"{len(result.sifted_alice)}")
            st.caption("Where bases matched")
        with m3:
            st.metric("Error rate", f"{result.error_rate:.2%}")
            st.caption("Estimated over sample")

        st.markdown("---")

        # Visuals
        v1, v2 = st.columns(2)
        with v1:
            st.subheader("Alice vs Bob (first 32)")
            preview_n = min(32, len(result.alice_bits))
            tbl = {
                "Alice bits": result.alice_bits[:preview_n],
                "Alice bases": result.alice_bases[:preview_n],
                "Bob bases": result.bob_bases[:preview_n],
                "Bob bits": result.bob_bits[:preview_n],
            }
            st.dataframe(tbl, use_container_width=True, hide_index=True)
        with v2:
            st.subheader("Basis Match Distribution")
            matches = sum(1 for a, b in zip(result.alice_bases, result.bob_bases) if a == b)
            fig2, ax2 = plt.subplots()
            ax2.bar(["Match", "Mismatch"], [matches, len(result.alice_bases) - matches])
            ax2.set_ylabel("Count")
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

        if show_bloch:
            st.subheader("Bloch Spheres (first 3 qubits)")
            plot_bloch_samples(result.alice_bits, result.alice_bases)

        # Final key block
        st.subheader("Final Secure Key")
        if result.final_key:
            key_str = ''.join(map(str, result.final_key))
            st.code(key_str[:256] + ("…" if len(key_str) > 256 else ""))
            buf = io.BytesIO(key_str.encode())
            st.download_button("Download Key", data=buf, file_name="bb84_key.txt", mime="text/plain")
        else:
            st.error("No secure key generated. Increase bits or check error rate.")

        # Notes
        with st.expander("What do the stats mean?"):
            st.markdown(
                """
                - **Sifted length**: positions where Alice and Bob chose the same basis.
                - **Error rate**: estimated by revealing a random subset of sifted bits; high error suggests eavesdropping or noise.
                - **Final key**: sifted bits with sampled positions removed.
                """
            )




def section_theory():
    # Inject CSS
    st.markdown("""
    <style>
        .theory-card {
            background: linear-gradient(135deg, 
                          rgba(38, 61, 235, 0.98) 0%, 
                          rgba(20, 25, 60, 0.98) 100%);
            border-radius: 16px;
            padding: 28px;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25),
                        0 0 0 1px rgba(113, 119, 255, 0.15),
                        inset 0 0 20px rgba(99, 102, 241, 0.1);
            border: 1px solid rgba(113, 119, 255, 0.2);
            margin-bottom: 2rem;
            position: relative;
            overflow: hidden;
        }
        
        .theory-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, 
                          transparent 0%, 
                          rgba(99, 102, 241, 0.6) 50%, 
                          transparent 100%);
        }
        
        .protocol-step {
            padding: 18px;
            margin: 14px 0;
            border-left: 4px solid #6366f1;
            background: rgba(30, 41, 59, 0.7);
            border-radius: 0 8px 8px 0;
            transition: all 0.3s ease;
            position: relative;
            backdrop-filter: blur(4px);
        }
        
        .protocol-step:hover {
            background: rgba(30, 41, 59, 0.9);
            transform: translateX(6px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
        }
        
        .protocol-step h4 {
            color: #e0e7ff;
            margin: 0 0 10px 0;
            font-size: 1.15rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .protocol-step b {
            background: linear-gradient(90deg, #818cf8 0%, #a855f7 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .basis-matrix {
            background: rgba(20, 30, 48, 0.8);
            border-radius: 8px;
            padding: 14px;
            font-family: 'Courier New', monospace;
            margin: 10px 0;
            border: 1px solid rgba(99, 102, 241, 0.2);
            color: #e2e8f0;
        }
        
        .quantum-formula {
            font-family: 'Cambria Math', serif;
            background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.15rem;
        }
        
        .highlight-box {
            margin-top: 28px;
            padding: 18px;
            background: linear-gradient(135deg, 
                          rgba(16, 185, 129, 0.15) 0%, 
                          rgba(6, 78, 59, 0.15) 100%);
            border-radius: 8px;
            border-left: 4px solid #10b981;
            position: relative;
            overflow: hidden;
        }
        
        .highlight-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 100%;
            background: linear-gradient(90deg, 
                          rgba(16, 185, 129, 0.05) 0%, 
                          transparent 50%);
            pointer-events: none;
        }
        
        .highlight-box h4 {
            color: #a7f3d0;
            margin: 0 0 12px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .quantum-icon {
            width: 24px;
            height: 24px;
            filter: drop-shadow(0 0 4px rgba(167, 243, 208, 0.3));
        }
    </style>
    """, unsafe_allow_html=True)

    # Use HTML instead of SVG icons (Streamlit strips inline SVG often)
    st.markdown("""
<div class="theory-card">
<h2 style="margin-top: 0; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px;">
    📡 The BB84 Quantum Key Distribution Protocol
</h2>
<p style="color: #475569; line-height: 1.6; font-size: 1.05rem;">
    Proposed by <b>Bennett</b> and <b>Brassard</b> in 1984, BB84 was the first quantum cryptography 
    protocol and remains the foundation of <b>Quantum Key Distribution (QKD)</b>.
</p>
<div class="protocol-step">
    <h4>1. Preparation</h4>
    Alice encodes classical bits into quantum states:
    <div class="basis-matrix">
        <b>Rectilinear (+):</b><br>
        0 → |0⟩ = <span class="quantum-formula">[1; 0]</span><br>
        1 → |1⟩ = <span class="quantum-formula">[0; 1]</span><br><br>
        <b>Diagonal (×):</b><br>
        0 → |+⟩ = <span class="quantum-formula">1/√2[1; 1]</span><br>
        1 → |−⟩ = <span class="quantum-formula">1/√2[1; -1]</span>
    </div>
</div>
<div class="protocol-step">
    <h4>2. Quantum Transmission</h4>
    Alice sends qubits over a quantum channel. Any eavesdropping disturbs the states 
    (<i>No-Cloning Theorem</i>).
</div>
<div class="protocol-step">
<h4>3. Measurement</h4>
    Bob measures in a random basis.<br>
    <div style="margin: 8px 0; padding: 8px; background: rgba(239, 246, 255, 0.5); border-radius: 8px;">
        <b>Matching basis:</b> 100% correct<br>
        <b>Mismatched basis:</b> 50% random
    </div>
</div>
<div class="protocol-step">
    <h4>4. Sifting</h4>
    They compare bases publicly and discard mismatched results.
</div>
<div class="protocol-step">
    <h4>5. Error Estimation</h4>
    A subset of bits is revealed to estimate QBER.
    <br><b>QBER < 25%:</b> Secure<br>
    <b>QBER ≥ 25%:</b> Eavesdropping likely
</div>
<div class="protocol-step">
    <h4>6. Key Distillation</h4>
    If secure, they apply:<br>
    • Error correction<br>
    • Privacy amplification
</div>
<div class="highlight-box">
<h4>🔑 Quantum Advantage</h4>
Security is guaranteed by physics, not math:<br>
• No-Cloning Theorem<br>
• Measurement Disturbance<br>
• Uncertainty Principle
</div>
</div>
    """, unsafe_allow_html=True)


def section_team():
    st.markdown(
        """
        <style>
            .team-card {
                background: rgba(255, 255, 255, 0.96);
                border-radius: 16px;
                padding: 28px;
                box-shadow: 0 8px 32px rgba(2, 6, 23, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.3);
                margin-bottom: 2rem;
            }
            .team-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                gap: 20px;
                margin-top: 24px;
            }
            .member-card {
                background: white;
                border-radius: 12px;
                padding: 24px;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                border: 1px solid rgba(226, 232, 240, 0.5);
                transition: all 0.3s ease;
                position: relative;
                overflow: hidden;
            }
            .member-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1);
            }
            .member-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                width: 4px;
                height: 100%;
                background: linear-gradient(to bottom, #6366f1, #8b5cf6);
            }
            .member-role {
                font-size: 1.1rem;
                font-weight: 600;
                color: #0f172a;
                margin-bottom: 8px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .member-bio {
                color: #475569;
                line-height: 1.5;
                font-size: 0.95rem;
                margin-bottom: 16px;
                min-height: 60px;
            }
            .skill-tag {
                display: inline-block;
                background: rgba(238, 242, 255, 0.8);
                color: #4338ca;
                padding: 4px 12px;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 500;
                margin-right: 8px;
                margin-bottom: 8px;
                transition: all 0.2s ease;
            }
            .skill-tag:hover {
                background: #6366f1;
                color: white;
                transform: scale(1.05);
            }
            .team-icon {
                width: 24px;
                height: 24px;
                color: #4338ca;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
<div class="team-card">
<h2 style="margin-top: 0; color: #0f172a; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px;">
<span style="display: flex; align-items: center; gap: 10px;">
<svg class="team-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M17 21V19C17 17.9391 16.5786 16.9217 15.8284 16.1716C15.0783 15.4214 14.0609 15 13 15H5C3.93913 15 2.92172 15.4214 2.17157 16.1716C1.42143 16.9217 1 17.9391 1 19V21" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M9 11C11.2091 11 13 9.20914 13 7C13 4.79086 11.2091 3 9 3C6.79086 3 5 4.79086 5 7C5 9.20914 6.79086 11 9 11Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M23 21V19C22.9993 18.1137 22.7044 17.2528 22.1614 16.5523C21.6184 15.8519 20.8581 15.3516 20 15.13" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M16 3.13C16.8604 3.3503 17.623 3.8507 18.1676 4.55231C18.7122 5.25392 19.0078 6.11683 19.0078 7.005C19.0078 7.89317 18.7122 8.75608 18.1676 9.45769C17.623 10.1593 16.8604 10.6597 16 10.88" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
Quantum Development Team
</span>
</h2>
<p style="color: #475569; line-height: 1.6;">
Our interdisciplinary team combines quantum physics expertise with software engineering and design 
to create accessible quantum education tools.
</p>
<div class="team-grid">
<div class="member-card">
<div class="member-role">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
Lead Quantum Developer
</div>
<div class="member-bio">
Designs and implements the quantum simulation engine, protocol logic, and visualization systems.
</div>
<div>
<span class="skill-tag">Python</span>
<span class="skill-tag">Qiskit</span>
<span class="skill-tag">Quantum Circuits</span>
<span class="skill-tag">Streamlit</span>
</div>
</div>

<div class="member-card">
<div class="member-role">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M19 21L12 16L5 21V5C5 4.46957 5.21071 3.96086 5.58579 3.58579C5.96086 3.21071 6.46957 3 7 3H17C17.5304 3 18.0391 3.21071 18.4142 3.58579C18.7893 3.96086 19 4.46957 19 5V21Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
Quantum Researcher
</div>
<div class="member-bio">
Ensures protocol accuracy and develops educational content about quantum information theory.
</div>
<div>
<span class="skill-tag">QKD Theory</span>
<span class="skill-tag">Quantum Info</span>
<span class="skill-tag">Pedagogy</span>
<span class="skill-tag">Security Analysis</span>
</div>
</div>

<div class="member-card">
<div class="member-role">
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
<path d="M14 2H6C5.46957 2 4.96086 2.21071 4.58579 2.58579C4.21071 2.96086 4 3.46957 4 4V20C4 20.5304 4.21071 21.0391 4.58579 21.4142C4.96086 21.7893 5.46957 22 6 22H18C18.5304 22 19.0391 21.7893 19.4142 21.4142C19.7893 21.0391 20 20.5304 20 20V8L14 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M14 2V8H20" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M16 13H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M16 17H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M10 9H9H8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
UX & Documentation
</div>
<div class="member-bio">
Crafts intuitive interfaces and learning materials to make quantum concepts accessible.
</div>
<div>
<span class="skill-tag">UI/UX</span>
<span class="skill-tag">Technical Writing</span>
<span class="skill-tag">Visual Design</span>
<span class="skill-tag">Accessibility</span>
</div>
</div>
</div>

<div style="margin-top: 24px; padding: 16px; background: rgba(240, 253, 250, 0.5); border-radius: 8px; border-left: 4px solid #10b981;">
<h4 style="margin: 0 0 8px 0; color: #047857;">Collaboration</h4>
<p style="margin: 0; color: #064e3b;">
We follow agile development principles with weekly syncs between quantum theory and 
implementation teams to ensure both technical accuracy and educational value.
</p>
</div>
</div>
""",
        unsafe_allow_html=True
    )

def section_faq():
    st.markdown(
        """
        <style>
            .faq-card {
                background: linear-gradient(135deg, rgba(10, 12, 28, 0.98) 0%, rgba(20, 25, 60, 0.98) 100%);
                border-radius: 16px;
                padding: 28px;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25),
                            0 0 0 1px rgba(113, 119, 255, 0.15),
                            inset 0 0 20px rgba(99, 102, 241, 0.1);
                border: 1px solid rgba(113, 119, 255, 0.2);
                margin-bottom: 2rem;
            }
            
            .faq-card h3 {
                color: white;
                margin-top: 0;
                margin-bottom: 24px;
                font-size: 1.6rem;
                background: linear-gradient(90deg, #646cff 0%, #a855f7 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                padding-bottom: 12px;
                border-bottom: 1px solid rgba(113, 119, 255, 0.3);
            }
            
            details {
                margin-bottom: 16px;
                border-radius: 8px;
                overflow: hidden;
                transition: all 0.3s ease;
                background: rgba(30, 41, 59, 0.7);
                border: 1px solid rgba(99, 102, 241, 0.2);
            }
            
            details[open] {
                background: rgba(30, 41, 59, 0.9);
                box-shadow: 0 4px 12px rgba(99, 102, 241, 0.15);
            }
            
            summary {
                padding: 18px;
                font-weight: 600;
                color: #e0e7ff;
                cursor: pointer;
                position: relative;
                list-style: none;
                transition: all 0.2s ease;
                font-size: 1.1rem;
            }
            
            summary:hover {
                color: white;
                background: rgba(99, 102, 241, 0.1);
            }
            
            summary::after {
                content: "+";
                position: absolute;
                right: 18px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 1.2rem;
                color: #818cf8;
                transition: all 0.3s ease;
            }
            
            details[open] summary::after {
                content: "-";
                color: #a855f7;
            }
            
            details div {
                padding: 0 18px 18px 18px;
                color: #cbd5e1;
                line-height: 1.6;
                font-size: 1rem;
                border-top: 1px solid rgba(99, 102, 241, 0.2);
                margin-top: 8px;
                padding-top: 12px;
            }
            
            .quantum-badge {
                display: inline-block;
                background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
                color: white;
                padding: 4px 10px;
                border-radius: 999px;
                font-size: 0.8rem;
                font-weight: 600;
                margin-right: 8px;
            }
        </style>
        
<div class="faq-card">
<h3>❓ Frequently Asked Questions</h3>

<details>
<summary><span class="quantum-badge">Visualization</span> Why do Bloch spheres sometimes look identical?</summary>
<div>
States like |0⟩ and |+⟩ may appear similar if the visualization library projects them similarly. 
This occurs because the Bloch sphere is a 2D representation of 3D quantum states. 
<br><br>
<b>Solution:</b> 
<ul>
<li>Ensure Qiskit visualization components are properly installed</li>
<li>View multiple qubits to compare different states</li>
<li>Rotate the visualization if interactive 3D is available</li>
</ul>
</div>
</details>

<details>
    <summary><span class="quantum-badge">Security</span> What threshold indicates eavesdropping?</summary>
    <div>
        In ideal BB84 protocol conditions:
        <ul>
            <li>An intercept-resend eavesdropper yields ~25% error rate</li>
            <li>Real-world quantum channels typically tolerate only 2-5% error</li>
            <li>Error rates above 10% usually indicate significant interference</li>
        </ul>
        <br>
        Our simulator shows real-time error rate analysis to help identify potential eavesdropping.
    </div>
</details>

<details>
    <summary><span class="quantum-badge">Reproducibility</span> Can I reproduce results?</summary>
    <div>
        Yes! For consistent, reproducible runs:
        <ol>
            <li>Set a fixed random seed in the Simulator parameters</li>
            <li>Use the same number of qubits for each run</li>
            <li>Maintain identical eavesdropper settings</li>
        </ol>
        <br>
        This ensures you get identical results for demonstration and debugging purposes.
    </div>
</details>

<details>
<summary><span class="quantum-badge">Technical</span> What are the system requirements?</summary>
<div>
<b>Minimum Requirements:</b>
<ul>
<li>Modern web browser (Chrome, Firefox, Edge)</li>
<li>Python 3.8+ for local deployment</li>
<li>4GB RAM for quantum simulations</li>
</ul>
<br>
<b>Recommended:</b>
<ul>
<li>8GB+ RAM for larger simulations</li>
<li>Qiskit installed for full visualization features</li>
<li>WebGL support for interactive 3D views</li>
</ul>
</div>
</details>
</div>
        """,
        unsafe_allow_html=True
    )


def section_contact():
    st.markdown(
        """
<div class="card">
<h3>Contact</h3>
<p class="muted">Have a feature request or found a bug? Drop a note below.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.form("contact"):
        name = st.text_input("Name")
        email = st.text_input("Email")
        message = st.text_area("Message")
        submitted = st.form_submit_button("Send")
        if submitted:
            if name and email and message:
                st.success("Thanks! We'll get back to you soon.")
            else:
                st.error("Please fill in all fields.")


def footer():
    st.markdown(
        """
<div class="footer">Built with ❤️ using Streamlit. Optional quantum backend powered by Qiskit Aer.</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# App Entrypoint
# ---------------------------



def main():
    header()
    st.set_page_config(
        page_title="BB84 Quantum Key Distribution",
        page_icon="🔐",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    current_page = sidebar_nav()

    # Page routing
    if current_page == "Home":
        section_home()
    elif current_page == "Simulator":
        section_simulator()
    elif current_page == "Theory":
        section_theory()
    elif current_page == "Team":
        section_team()
    elif current_page == "FAQ":
        section_faq()
    elif current_page == "Contact":
        section_contact()

    footer()


if __name__ == "__main__":
    main()
