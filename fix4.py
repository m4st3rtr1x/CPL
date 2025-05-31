import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import io

st.set_page_config(page_title="Clustering CPL-PL", layout="wide")
st.title("Analisis Clustering CPL & PL berdasarkan Nilai A Mahasiswa")

# Upload file Excel
uploaded_file = st.file_uploader("Unggah file Excel", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Rename kolom agar sesuai
    df = df.rename(columns={
        'kodematakuliah': 'Kode Mata Kuliah',
        'NAMA MATAKULIAH': 'Nama Mata Kuliah',
        'CPL Yang dipenuhi': 'CPL',
        'PL yang dipenuhi': 'PL',
        'nilai': 'Nilai'
    })

    # Pra-pemrosesan
    df = df.dropna()
    df = df.drop_duplicates()
    df = df[df['Nilai'] == 'A']

    excluded_courses = [
        'AGAMA', 'BAHASA INDONESIA', 'BAHASA INGGRIS',
        'KEWIRAUSAHAAN', 'HARDWARE KOMPUTER', 'ANIMASI DAN MULTIMEDIA'
    ]
    df = df[~df['Nama Mata Kuliah'].str.upper().isin(excluded_courses)]

    # Pilih kolom penting
    df = df[['Kode Mata Kuliah', 'Nama Mata Kuliah', 'CPL', 'PL']]

    # Encoding
    le_nama_mk = LabelEncoder()
    le_cpl = LabelEncoder()
    le_pl = LabelEncoder()

    df_encoded = df.copy()
    df_encoded['Nama Mata Kuliah'] = le_nama_mk.fit_transform(df_encoded['Nama Mata Kuliah'])
    df_encoded['CPL'] = le_cpl.fit_transform(df_encoded['CPL'])
    df_encoded['PL'] = le_pl.fit_transform(df_encoded['PL'])

    # Agregasi
    df_encoded['Jumlah Mahasiswa'] = 1
    df_agg = df_encoded.groupby(['Kode Mata Kuliah', 'Nama Mata Kuliah', 'CPL', 'PL']).agg({
        'Jumlah Mahasiswa': 'sum'
    }).reset_index()

    # Normalisasi
    features = ['Nama Mata Kuliah', 'CPL', 'PL', 'Jumlah Mahasiswa']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_agg[features])

    # Elbow method
    inertia = []
    K = range(1, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)

    st.subheader("Elbow Method")
    fig, ax = plt.subplots()
    ax.plot(K, inertia, marker='o')
    ax.set_xlabel('Jumlah Cluster (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Menentukan k Optimal')
    st.pyplot(fig)

    # Input jumlah cluster dari user
    k_opt = st.slider("Pilih jumlah cluster", min_value=2, max_value=10, value=4)

    # Clustering
    kmeans = KMeans(n_clusters=k_opt, random_state=42)
    df_agg['Cluster'] = kmeans.fit_predict(X_scaled)

    # Decode
    df_agg['Nama Mata Kuliah'] = le_nama_mk.inverse_transform(df_agg['Nama Mata Kuliah'])
    df_agg['CPL'] = le_cpl.inverse_transform(df_agg['CPL'])
    df_agg['PL'] = le_pl.inverse_transform(df_agg['PL'])

    # Interpretasi & Rekomendasi (contoh statis untuk 4 cluster)
    interpretasi_map = {
        0: 'Cluster ini menunjukkan mahasiswa dengan kemampuan praktik langsung yang cukup baik dan kecenderungan pada penguasaan perangkat keras serta konsultansi di bidang Teknologi Informasi. Mata kuliah seperti IMK, Kerja Praktek dan Manajemen Proyek mengindikasikan mahasiswa memahami bagaimana merancang interaksi sistem berbasis perangkat fisik dengan melalui proses manajemen proyek teknologi.',
        1: 'Cluster ini menunjukkan mahasiswa memiliki kompetensi awal dalam pemrograman dan pengelolaan data. Mata kuliah seperti Statistika dan Probabilitas mendukung fondasi pemahaman manajemen serta struktur data, IMK dan Manajemen Rantai Pasok mendukung fondasi pemahaman terhadap logika',
        2: 'Cluster ini berisi mahasiswa yang kuat secara sistemik dan manajerial, serta memiliki kecakapan teknis dalam aspek operasional sistem dan perangkat keras. Dominasi mata kuliah sistem informasi, sistem operasi dan riset operasional menunjukkan pemahaman mahasiswa terkait dengan keterlibatan pada proyek perencanaan dan efisiensi teknologi. Mirip dengan Cluster 0 dari sisi dominasi PL, tetapi Cluster 2 memiliki kapasitas mahasiswa yang lebih tinggi dan lebih kuat di aspek sistemik.',
        3: 'Cluster paling unggul secara performa mahasiswa, ditandai dengan profil lulusan yang fokus pada data science dan pengelolaan data. Mahasiswa dalam kelompok ini menunjukkan penguasaan analisis, pemodelan sistem, dan eksplorasi data mendalam.Cluster ini bisa menjadi target utama pengembangan lanjutan program kekhasan Data Science, dengan pembekalan tambahan pada PL6 (AI) agar lengkap dalam alur Data Science > Machine Learning.'
    }
    profesi_map = {
        0: 'Hardware Integrator & Konsultan Teknologi, System Analyst untuk sistem berbasis perangkat fisik, IT Consultant untuk pengembangan solusi berbasis IoT atau sistem digital',
        1: 'Junior Data Engineer / Junior DBA, Frontend Developer dengan basis data, Entry-Level Software Developer',
        2: 'IT Infrastructure Consultant, System Administrator, Technical Advisor untuk Sistem Informasi',
        3: 'Data Scientist, Business Intelligence Specialist, Predictive Analyst, Big Data Analyst'
    }

    df_agg['Interpretasi'] = df_agg['Cluster'].map(interpretasi_map)
    df_agg['Rekomendasi Profesi'] = df_agg['Cluster'].map(profesi_map)

    # Ringkasan cluster
    cluster_summary = df_agg.groupby('Cluster').agg({
        'Jumlah Mahasiswa': 'mean',
        'Nama Mata Kuliah': lambda x: ', '.join(x.unique()[:3]),
        'CPL': lambda x: ', '.join(x.unique()[:3]),
        'PL': lambda x: ', '.join(x.unique()[:2]),
        'Interpretasi': 'first',
        'Rekomendasi Profesi': 'first'
    }).reset_index()

    cluster_summary.rename(columns={
        'Nama Mata Kuliah': 'Mata Kuliah',
        'CPL': 'CPL',
        'PL': 'PL Dominan',
        'Interpretasi': 'Interpretasi',
        'Rekomendasi Profesi': 'Rekomendasi Profesi',
        'Jumlah Mahasiswa': 'Rata-rata Mahasiswa'
    }, inplace=True)

    st.subheader("Ringkasan Cluster")
    st.dataframe(cluster_summary)

    # Download hasil ringkasan cluster
    towrite = io.BytesIO()
    cluster_summary.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)

    st.download_button(
        label="Download Hasil Ringkasan",
        data=towrite,
        file_name="hasil_ringkasan_cluster.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.subheader("Data Lengkap per Record")
    st.dataframe(df_agg)

    # Download hasil per record
    towrite2 = io.BytesIO()
    df_agg.to_excel(towrite2, index=False, engine='openpyxl')
    towrite2.seek(0)

    st.download_button(
        label="Download Hasil Per Record",
        data=towrite2,
        file_name="hasil_per_record_cluster.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
