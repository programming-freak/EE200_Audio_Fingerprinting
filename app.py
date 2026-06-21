import tempfile
import pandas as pd
import streamlit as st
from matcher import (
    identify_song,
    create_spectrogram,
    create_constellation,
    create_offset_histogram
)
# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Audio Fingerprinting System",
    layout="wide"
)
# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

.stApp{
    background-color:#050816;
}

[data-testid="stSidebar"]{
    background-color:#0B1023;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white !important;
}

div[data-testid="stMetric"]{
    background:rgba(255,255,255,0.05);
    border-radius:15px;
    padding:15px;
}

</style>
""", unsafe_allow_html=True)
# =====================================================
# HEADER
# =====================================================

st.markdown("""
<h1 style="
text-align:center;
font-size:55px;
font-weight:700;
color:white;
">
🎵 Audio Fingerprinting System
</h1>

<p style="
text-align:center;
font-size:20px;
color:#00FFFF;
">
Shazam-Inspired Song Recognition using Audio Fingerprints
</p>
""", unsafe_allow_html=True)

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
# 🎵 Music ID

Audio Fingerprinting Engine
""")

mode = st.sidebar.radio(
    "Mode",
    [
        "Single Clip",
        "Batch Mode"
    ]
)

# =====================================================
# SINGLE CLIP MODE
# =====================================================

if mode == "Single Clip":

    uploaded_file = st.file_uploader(
        "Upload MP3",
        type=["mp3"]
    )

    if uploaded_file:

        st.audio(uploaded_file)

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp3"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            temp_path = tmp.name

        with st.spinner(
            "Analyzing audio fingerprint..."
        ):

            song, confidence = identify_song(
                temp_path
            )

        # ==========================================
        # PREDICTION CARD
        # ==========================================

        st.markdown(f"""
        <div style="
        padding:25px;
        border-radius:20px;
        background:linear-gradient(
        90deg,
        rgba(0,255,255,0.08),
        rgba(128,0,255,0.08)
        );
        border:1px solid rgba(0,255,255,0.2);
        margin-bottom:20px;
        ">

        <h4 style="color:#00FFFF;">
        Predicted Song
        </h4>

        <h2 style="color:white;">
        {song}
        </h2>

        <p style="color:#00FFAA;">
        Confidence Score: {confidence}
        </p>

        </div>
        """, unsafe_allow_html=True)

        # ==========================================
        # STATS
        # ==========================================

        c1, c2 = st.columns(2)

        c1.metric(
            "Confidence",
            confidence
        )

        c2.metric(
            "Prediction",
            song
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ==========================================
        # SPECTROGRAM + CONSTELLATION
        # ==========================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("-> Spectrogram")

            fig = create_spectrogram(temp_path)
            st.write("Figure created")
            st.pyplot(fig)
            st.write("Figure rendered")

        with col2:

            st.subheader("-> Constellation Map")

            st.pyplot(
                create_constellation(
                    temp_path
                )
            )

        # ==========================================
        # OFFSET HISTOGRAM
        # ==========================================

        st.subheader(
            "-> Offset Histogram"
        )

        st.pyplot(
            create_offset_histogram(
                temp_path
            )
        )

# =====================================================
# BATCH MODE
# =====================================================

if mode == "Batch Mode":

    st.markdown("""
    <h2 style="color:white;">
    🎵 Identify Many Clips At Once
    </h2>

    <p style="color:#AAAAAA;">
    Upload multiple audio clips and identify them
    against the indexed song database.
    </p>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload Query Clips",
        type=["mp3"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.success(
            f"{len(uploaded_files)} file(s) selected"
        )

        # ==========================================
        # RUN BATCH BUTTON
        # ==========================================

        run_batch = st.button(
            "-> Run Batch",
            use_container_width=False
        )

        if run_batch:

            results = []

            progress_bar = st.progress(0)

            status_text = st.empty()

            total = len(uploaded_files)

            for idx, file in enumerate(uploaded_files):

                status_text.markdown(
                    f"""
                    <span style='color:#00FFFF'>
                    Identifying ...
                    {idx+1}/{total}
                    </span>
                    """,
                    unsafe_allow_html=True
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3"
                ) as tmp:

                    tmp.write(
                        file.read()
                    )

                    temp_path = tmp.name

                song, confidence = identify_song(
                    temp_path
                )

                results.append(
                    [
                        file.name,
                        song,
                        confidence
                    ]
                )

                progress_bar.progress(
                    (idx + 1) / total
                )

            status_text.success(
                "Batch Processing Complete!"
            )

            df = pd.DataFrame(
                results,
                columns=[
                    "filename",
                    "prediction",
                    "confidence"
                ]
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "📥 Download Results CSV",
                df.to_csv(index=False),
                "results.csv",
                "text/csv"
            )


