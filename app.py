import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🎬 Movie Recommender System", layout="wide")

# --- HEADER VISUAL ---
st.markdown("""
    <style>
        .main {background-color: #f9fafb;}
        .block-container {padding-top: 2rem; display: flex; justify-content: center; align-items: center; flex-direction: column;}
        .title {
            color: #292968; 
            font-size: 3.6rem;    /* Ukuran diperbesar */
            font-weight: bold; 
            text-align: center;
            font-family: 'Segoe UI', Arial, sans-serif; 
            letter-spacing: 1px;
        }
        .subtitle {color: #6366f1; font-size: 1.35rem; text-align: center;}
        .header-art {
            width: 100%;
            height: 190px;              /* Lebih tinggi */
            padding-top: 30px;          /* Tambah padding atas */
            padding-bottom: 30px;       /* Tambah padding bawah */
            background: linear-gradient(90deg, #6366f1 0%, #fbbf24 100%);
            border-radius: 25px;        /* Biar makin soft */
            margin-bottom: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 6px 32px #6366f133;
        }

        .emoji {font-size: 3.2rem; margin: 0 1.6rem;}
        .fav-label {font-weight: bold; font-size: 1.2rem; color: #222; margin-bottom: 0.7em;}

        /* CSS for centering the table */
        .cool-table {
            width: 95%; max-width: 1000px;
            margin: 32px auto 0 auto;
            background: #f8faff;
            border-radius: 18px;
            box-shadow: 0 2px 24px #d1e3fd77;
            overflow-x: auto;
            text-align: left;  /* Align content to the left */
        }

        .cool-table table {
            width: 100%; border-collapse: separate; border-spacing: 0;
            background: transparent;
            font-size: 1.10rem;
            border-radius: 17px;
            overflow: hidden;
            margin: auto;
        }

        .cool-table th, .cool-table td {
            padding: 15px 16px;
            text-align: left;  /* Left-align text in table cells */
            border: none;
        }

        .cool-table th {
            position: sticky; top: 0; z-index: 2;
            background: linear-gradient(90deg, #edf1fd 85%, #ebeafd 100%);
            color: #3637a6;
            font-size: 1.11em;
            font-weight: bold;
            border-bottom: 2.5px solid #e3e6f8;
            letter-spacing: .4px;
            box-shadow: 0 1px 0 #e3e6f8;
        }

        .cool-table tr {transition: background .17s;}
        .cool-table tr:hover td {background: #e7ecff !important;}
        .cool-table td {background: #fafaff;}
        .cool-table tr:nth-child(even) td {background: #f6f7fe;}
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    with open('movie_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['df'], data['tfidf'], data['tfidf_matrix']

df, tfidf, tfidf_matrix = load_data()

# Visual header: emoji popcorn, clapboard, ticket
st.markdown("""
    <style>
        .title {
            color: #292968;
            font-size: 2.4rem;
            font-weight: bold;
            text-align: center;
            font-family: 'Segoe UI', Arial, sans-serif;
            letter-spacing: 1px;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    <div class="header-art">
        <span class="emoji">🍿</span>
        <span class="title">Movie Recommender System</span>
        <span class="emoji">🎟️</span>
    </div>
    """, unsafe_allow_html=True
)

st.markdown('<div class="subtitle">Temukan rekomendasi film berdasarkan film yang kamu input!</div>', unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

top_fav = df.reset_index()['title'].tolist()

# INIT SESSION STATE
if 'film_input_manual' not in st.session_state:
    st.session_state['film_input_manual'] = ""
if 'selectbox_favorite' not in st.session_state:
    st.session_state['selectbox_favorite'] = ""
if 'input_mode' not in st.session_state:
    st.session_state['input_mode'] = "Manual"

# ------------- INPUT SECTION -------------
col_main, col_fav = st.columns([1, 1], gap="large")  # Equal column width

with col_main:
    # MODE INPUT (radio) di atas input
    input_mode = st.radio(
        "Pilih metode input film:",
        ("Manual", "Pilih dari Film Populer"),
        index=0 if st.session_state['input_mode'] == "Manual" else 1,
        key="input_mode"
    )
    st.markdown('<div style="margin-bottom:0.7em;"></div>', unsafe_allow_html=True)

    # Dapatkan value input yang akan dipakai (tidak update session_state manual!)
    if input_mode == "Manual":
        film_manual = st.text_input(
            "🎬 Tulis judul film:",
            value=st.session_state['film_input_manual'],
            key="film_input_manual",
            disabled=False
        )
        film_selected = film_manual
    else:
        film_manual = st.text_input(
            "🎬 Tulis judul film:",
            value=st.session_state['film_input_manual'],
            key="film_input_manual",
            disabled=True
        )
        film_selected = st.session_state['selectbox_favorite']

with col_fav:
    st.markdown('<span class="fav-label">⭐ Film Populer</span>', unsafe_allow_html=True)
    favorite_selected = st.selectbox(
        "Pilih berdasarkan film populer yang ada di dataset kami:",
        [""] + top_fav,
        key="selectbox_favorite",
        disabled=(input_mode == "Manual"),
        label_visibility="visible"
    )

# Separate slider and button in a new layout, matching the width of the inputs
col_slider_button = st.container()
with col_slider_button:
    # Set the slider and button within the same width as the input fields
    top_n = st.slider("🔎 Jumlah rekomendasi yang ingin ditampilkan", 1, 20, 10, key="slider_rek")
    btn = st.button("✨ Tampilkan Rekomendasi Film", use_container_width=True)

# Function to render table of recommendations without the percentage bar
def render_wide_table(result):
    html = """
    <div class="cool-table">
      <table>
        <thead>
          <tr>
            <th style='width:44px;text-align:center;'>No</th>
            <th style='min-width:250px;'>Judul Film</th>
          </tr>
        </thead>
        <tbody>
    """
    for idx, row in result.iterrows():
        html += f"<tr>"
        html += f"<td style='text-align:center;font-weight:600;'>{idx}</td>"
        # JUDUL FILM TEBAL
        html += f"<td style='font-weight:700;color:#26266d'>{row['title']}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    st.markdown(html, unsafe_allow_html=True)

if btn:
    movie_list_lower = [judul.lower() for judul in df.index]
    try:
        film_selected_val = film_selected or ""
        idx = movie_list_lower.index(film_selected_val.strip().lower())
        real_judul = df.index[idx]
        cos_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim_indices = cos_sim.argsort()[::-1][1:top_n+1]
        result = df.iloc[sim_indices].copy()
        result['similarity'] = cos_sim[sim_indices]
        result = result.reset_index()[['title', 'similarity']]
        result.index = range(1, len(result)+1)

        st.markdown(
            f"<div style='background-color:#d9f7e9;padding:14px;border-radius:10px;text-align:center;font-weight:bold;'>"
            f"Menampilkan {len(result)} rekomendasi terbaik untuk: <span style='color:#1f2937'>{real_judul}</span>"
            "</div>", unsafe_allow_html=True)
        st.write("")
        render_wide_table(result)
    except ValueError:
        st.error("Judul film tidak ditemukan di database. Silakan ketik judul yang benar atau pilih dari daftar favorit di kanan.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;font-size:18px;'>Created by Raihan Musyaffa Hanif | 51422357</div>", unsafe_allow_html=True)
