import os
import pickle
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.signal import spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict

# =====================================================
# LOAD DATABASE
# =====================================================

with open("database.pkl", "rb") as f:
    database = pickle.load(f)
# database={}
# =====================================================
# AUDIO LOADING
# =====================================================

def load_audio(path):

    audio, fs = librosa.load(
        path,
        sr=22050,
        mono=True
    )

    return audio, fs

# =====================================================
# SPECTROGRAM
# =====================================================

def compute_spectrogram(audio, fs):

    f, t, Sxx = spectrogram(
        audio,
        fs=fs,
        nperseg=2048,
        noverlap=1024
    )

    return f, t, Sxx

# =====================================================
# PEAK EXTRACTION
# =====================================================

def extract_peaks(
    Sxx,
    percentile=99.95
):

    threshold = np.percentile(
        Sxx,
        percentile
    )

    local_max = (
        Sxx ==
        maximum_filter(
            Sxx,
            size=(15,15)
        )
    )

    peaks = np.argwhere(
        local_max &
        (Sxx > threshold)
    )

    return peaks

# =====================================================
# HASH GENERATION
# =====================================================

def generate_hashes(
    peaks,
    fanout=5
):

    hashes = []

    for i in range(len(peaks)):

        for j in range(1, fanout + 1):

            if i+j >= len(peaks):
                break

            f1 = peaks[i][0]
            t1 = peaks[i][1]

            f2 = peaks[i+j][0]
            t2 = peaks[i+j][1]

            dt = t2 - t1

            hashes.append(
                (
                    f1,
                    f2,
                    dt,
                    t1
                )
            )

    return hashes

# =====================================================
# QUERY HASHES
# =====================================================

def get_hashes(audio_path):

    audio, fs = load_audio(
        audio_path
    )

    f, t, Sxx = compute_spectrogram(
        audio,
        fs
    )

    peaks = extract_peaks(
        Sxx
    )

    hashes = generate_hashes(
        peaks
    )

    return hashes

# =====================================================
# MATCHING
# =====================================================

def match_song(query_hashes):

    votes = defaultdict(int)

    for h in query_hashes:

        key = h[:3]

        query_time = h[3]

        if key not in database:
            continue

        for song_name, db_time in database[key]:

            offset = db_time - query_time

            votes[
                (song_name, offset)
            ] += 1

    return votes

# =====================================================
# IDENTIFICATION
# =====================================================

def identify_song(audio_path):

    query_hashes = get_hashes(
        audio_path
    )

    votes = match_song(
        query_hashes
    )

    if len(votes) == 0:

        return (
            "No Match Found",
            0
        )

    best_match = max(
        votes,
        key=votes.get
    )

    song_name = best_match[0]

    confidence = votes[
        best_match
    ]

    return (
        song_name,
        confidence
    )

# =====================================================
# BATCH PREDICTION
# =====================================================

def batch_predict(folder):

    results = []

    for file in os.listdir(folder):

        if not file.endswith(".mp3"):
            continue

        path = os.path.join(
            folder,
            file
        )

        prediction, confidence = identify_song(
            path
        )

        results.append(
            [
                file,
                prediction,
                confidence
            ]
        )

    return pd.DataFrame(
        results,
        columns=[
            "filename",
            "prediction",
            "confidence"
        ]
    )

# =====================================================
# SPECTROGRAM
# =====================================================

def create_spectrogram(audio_path):
    audio, fs = load_audio(audio_path)
    f, t, Sxx = compute_spectrogram(
        audio,
        fs
    )
    peaks = extract_peaks(
        Sxx
    )
    st.write("Peaks:", len(peaks))
    # Optional: limit peaks for cleaner visualization
    if len(peaks) > 1000:
        peaks = peaks[:1000]

    fig, ax = plt.subplots(
        figsize=(8,4),
        facecolor="#050816"
    )

    ax.set_facecolor("#050816")

    # Spectrogram
    ax.pcolormesh(
        t,
        f,
        10*np.log10(Sxx + 1e-12),
        shading="gouraud",
        cmap="magma"
    )

    # Peak markers
    ax.scatter(
        t[peaks[:,1]],
        f[peaks[:,0]],
        s=25,
        c="#00FFFF",
        edgecolors="white",
        linewidths=0.5,
        alpha=1.0,
        zorder=100
    )

    ax.set_title(
        f"Spectrogram with {len(peaks)} Peaks",
        color="#00FFFF",
        fontsize=14
    )

    ax.set_xlabel(
        "Time (s)",
        color="#00FFFF"
    )

    ax.set_ylabel(
        "Frequency (Hz)",
        color="#00FFFF"
    )

    ax.tick_params(
        colors="#00FFFF"
    )

    for spine in ax.spines.values():
        spine.set_color("#00FFFF")

    return fig

# =====================================================
# CONSTELLATION MAP
# =====================================================

def create_constellation(audio_path):

    audio, fs = load_audio(audio_path)

    f, t, Sxx = compute_spectrogram(
        audio,
        fs
    )

    peaks = extract_peaks(Sxx)

    fig, ax = plt.subplots(
        figsize=(8,4),
        facecolor="#050816"
    )

    ax.set_facecolor("#050816")

    ax.scatter(
        t[peaks[:,1]],
        f[peaks[:,0]],
        s=6,
        c="#00FFFF",
        alpha=0.85
    )

    ax.set_title(
        "Constellation Map",
        color="#00FFFF",
        fontsize=14
    )

    ax.set_xlabel(
        "Time (s)",
        color="#00FFFF"
    )

    ax.set_ylabel(
        "Frequency (Hz)",
        color="#00FFFF"
    )

    ax.tick_params(
        colors="#00FFFF"
    )

    for spine in ax.spines.values():
        spine.set_color("#00FFFF")

    return fig


# =====================================================
# OFFSET HISTOGRAM
# =====================================================

def create_offset_histogram(audio_path):

    query_hashes = get_hashes(
        audio_path
    )

    votes = match_song(
        query_hashes
    )

    best_song = max(
        votes,
        key=votes.get
    )[0]

    winning_offsets = []

    for (song, offset), count in votes.items():

        if song == best_song:

            winning_offsets.extend(
                [offset] * count
            )

    fig, ax = plt.subplots(
        figsize=(10,4),
        facecolor="#050816"
    )

    ax.set_facecolor("#050816")

    counts, bins, patches = ax.hist(
        winning_offsets,
        bins=100,
        color="#00FFFF",
        alpha=0.9
    )

    best_offset = bins[np.argmax(counts)]

    ax.axvline(
        best_offset,
        color="white",
        linestyle="--",
        linewidth=2,
        label=f"Best Offset: {int(best_offset)}"
    )

    ax.legend()

    ax.set_title(
        f"Offset Histogram ({best_song})",
        color="#00FFFF",
        fontsize=14
    )

    ax.set_xlabel(
        "Offset",
        color="#00FFFF"
    )

    ax.set_ylabel(
        "Frequency",
        color="#00FFFF"
    )

    ax.tick_params(
        colors="#00FFFF"
    )

    for spine in ax.spines.values():
        spine.set_color("#00FFFF")

    return fig
