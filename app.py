import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Pengaturan konfigurasi halaman
st.set_page_config(
    page_title="Analisis Penyewaan Sepeda",
    page_icon="🚴",
    layout="wide"
)

# Fungsi untuk mempersiapkan dataset
@st.cache_data
def prepare_dataset():
    # Load data dari file CSV
    daily_data = pd.read_csv("dataset/day.csv")
    hourly_data = pd.read_csv("dataset/hour.csv")
    
    # Menghapus kolom yang tidak diperlukan
    daily_data = daily_data.drop('workingday', axis=1)
    hourly_data = hourly_data.drop('workingday', axis=1)
    
    # Konversi tipe data tanggal
    daily_data['dteday'] = pd.to_datetime(daily_data['dteday'])
    hourly_data['dteday'] = pd.to_datetime(hourly_data['dteday'])
    
    # Konversi kolom kategorikal
    categorical_cols = ['season', 'mnth', 'holiday', 'weekday', 'weathersit']
    for col in categorical_cols:
        daily_data[col] = daily_data[col].astype('category')
        hourly_data[col] = hourly_data[col].astype('category')
    
    # Mengubah nama kolom untuk meningkatkan keterbacaan
    column_rename_map = {
        'yr': 'tahun',
        'mnth': 'bulan', 
        'hr': 'jam',
        'weekday': 'hari',
        'weathersit': 'kondisi_cuaca',
        'temp': 'temperatur',
        'atemp': 'temp_terasa',
        'hum': 'kelembaban',
        'windspeed': 'kecepatan_angin',
        'cnt': 'total_sewa'
    }
    
    daily_data = daily_data.rename(columns=column_rename_map)
    hourly_data = hourly_data.rename(columns=column_rename_map)
    
    # Memetakan nilai numerik ke nama yang sesuai
    # Mapping untuk musim
    season_map = {1: 'Musim Semi', 2: 'Musim Panas', 3: 'Musim Gugur', 4: 'Musim Dingin'}
    daily_data['season'] = daily_data['season'].map(season_map)
    hourly_data['season'] = hourly_data['season'].map(season_map)
    
    # Mapping untuk bulan
    bulan_map = {
        1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 
        5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus', 
        9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
    }
    daily_data['bulan'] = daily_data['bulan'].map(bulan_map)
    hourly_data['bulan'] = hourly_data['bulan'].map(bulan_map)
    
    # Mapping untuk kondisi cuaca
    cuaca_map = {
        1: 'Cerah', 
        2: 'Berkabut', 
        3: 'Hujan/Salju Ringan', 
        4: 'Hujan/Salju Deras'
    }
    daily_data['kondisi_cuaca'] = daily_data['kondisi_cuaca'].map(cuaca_map)
    hourly_data['kondisi_cuaca'] = hourly_data['kondisi_cuaca'].map(cuaca_map)
    
    # Mapping untuk hari dalam seminggu
    hari_map = {
        0: 'Minggu', 
        1: 'Senin', 
        2: 'Selasa', 
        3: 'Rabu',
        4: 'Kamis', 
        5: 'Jumat', 
        6: 'Sabtu'
    }
    daily_data['hari'] = daily_data['hari'].map(hari_map)
    hourly_data['hari'] = hourly_data['hari'].map(hari_map)
    
    # Mapping untuk tahun
    daily_data['tahun'] = daily_data['tahun'].map({0: '2011', 1: '2012'})
    hourly_data['tahun'] = hourly_data['tahun'].map({0: '2011', 1: '2012'})
    
    # Membuat kolom hari kerja/libur
    def kategorikan_hari(hari):
        return "Akhir Pekan" if hari in ["Sabtu", "Minggu"] else "Hari Kerja"
    
    daily_data['kategori_hari'] = daily_data['hari'].apply(kategorikan_hari)
    hourly_data['kategori_hari'] = hourly_data['hari'].apply(kategorikan_hari)
    
    # Mengelompokkan kelembaban
    def kategorikan_kelembaban(kelembaban):
        if kelembaban < 0.4:
            return "Sangat Kering"
        elif kelembaban < 0.6:
            return "Normal"
        elif kelembaban < 0.8:
            return "Lembab"
        else:
            return "Sangat Lembab"
    
    daily_data['kategori_kelembaban'] = daily_data['kelembaban'].apply(kategorikan_kelembaban)
    hourly_data['kategori_kelembaban'] = hourly_data['kelembaban'].apply(kategorikan_kelembaban)
    
    return daily_data, hourly_data

# --- Fungsi-fungsi untuk analisis data ---
def analisis_jam(data_jam):
    """Menganalisis pola penyewaan sepeda berdasarkan jam"""
    hasil_analisis = data_jam.groupby('jam')['total_sewa'].sum().reset_index()
    return hasil_analisis

def analisis_musim(data_harian):
    """Menganalisis pola penyewaan sepeda berdasarkan musim"""
    hasil_analisis = data_harian.groupby('season')['total_sewa'].sum().reset_index()
    hasil_analisis = hasil_analisis.sort_values('total_sewa', ascending=False)
    return hasil_analisis

def analisis_cuaca(data_jam):
    """Menganalisis pola penyewaan sepeda berdasarkan kondisi cuaca"""
    hasil_analisis = data_jam.groupby('kondisi_cuaca')['total_sewa'].sum().reset_index()
    hasil_analisis = hasil_analisis.sort_values('total_sewa', ascending=False)
    return hasil_analisis

def analisis_kategori_hari(data_harian):
    """Menganalisis perbedaan pola penyewaan antara hari kerja dan akhir pekan"""
    hasil_analisis = data_harian.groupby('kategori_hari')['total_sewa'].agg(['sum', 'mean']).reset_index()
    hasil_analisis.columns = ['kategori_hari', 'total_sewa', 'rata_rata_sewa']
    return hasil_analisis

def analisis_tren_bulanan(data_harian):
    """Menganalisis tren bulanan penyewaan sepeda"""
    hasil_analisis = data_harian.groupby(pd.Grouper(key='dteday', freq='M')).agg({
        'total_sewa': 'sum',
        'casual': 'sum',
        'registered': 'sum'
    }).reset_index()
    return hasil_analisis

def analisis_segmentasi_pelanggan(data_harian):
    """Menganalisis distribusi antara pelanggan casual dan registered"""
    total_casual = data_harian['casual'].sum()
    total_registered = data_harian['registered'].sum()
    
    hasil_analisis = pd.DataFrame({
        'jenis_pelanggan': ['Pengguna Casual', 'Pengguna Terdaftar'],
        'jumlah': [total_casual, total_registered]
    })
    return hasil_analisis

def analisis_kelembaban(data_jam):
    """Menganalisis pengaruh kelembaban terhadap penyewaan sepeda"""
    hasil_analisis = data_jam.groupby('kategori_kelembaban').agg({
        'total_sewa': ['sum', 'mean']
    }).reset_index()
    hasil_analisis.columns = ['kategori_kelembaban', 'total_sewa', 'rata_rata_sewa']
    return hasil_analisis

# Load dan persiapkan data
data_harian, data_jam = prepare_dataset()

# --- DASHBOARD UI ---
st.title("🚴 Analisis Penyewaan Sepeda")
st.markdown("Dashboard interaktif untuk mengeksplorasi pola penyewaan sepeda dari layanan bike sharing")

# --- FILTERS ---
st.sidebar.title("Filter Data")

# Filter tahun
tahun_opsi = st.sidebar.radio(
    "Pilih Tahun", 
    options=["Semua Tahun", "2011", "2012"]
)

if tahun_opsi != "Semua Tahun":
    filtered_daily = data_harian[data_harian['tahun'] == tahun_opsi]
    filtered_hourly = data_jam[data_jam['tahun'] == tahun_opsi]
else:
    filtered_daily = data_harian
    filtered_hourly = data_jam

# Filter musim
musim_opsi = st.sidebar.multiselect(
    "Pilih Musim",
    options=data_harian['season'].unique(),
    default=data_harian['season'].unique()
)

filtered_daily = filtered_daily[filtered_daily['season'].isin(musim_opsi)]
filtered_hourly = filtered_hourly[filtered_hourly['season'].isin(musim_opsi)]

# Filter kategori hari
kategori_hari_opsi = st.sidebar.multiselect(
    "Pilih Kategori Hari",
    options=data_harian['kategori_hari'].unique(),
    default=data_harian['kategori_hari'].unique()
)

filtered_daily = filtered_daily[filtered_daily['kategori_hari'].isin(kategori_hari_opsi)]
filtered_hourly = filtered_hourly[filtered_hourly['kategori_hari'].isin(kategori_hari_opsi)]

# --- METRICS SUMMARY ---
st.header("Ringkasan Metrik")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_penyewaan = int(filtered_daily['total_sewa'].sum())
    st.metric("Total Penyewaan", f"{total_penyewaan:,}")

with col2:
    rata_penyewaan = int(filtered_daily['total_sewa'].mean())
    st.metric("Rata-rata Harian", f"{rata_penyewaan:,}")

with col3:
    max_info = filtered_daily.loc[filtered_daily['total_sewa'].idxmax()]
    st.metric(
        "Penyewaan Tertinggi", 
        f"{int(max_info['total_sewa']):,}",
        f"{max_info['dteday'].strftime('%d %b %Y')}"
    )

with col4:
    hari_dalam_data = filtered_daily.shape[0]
    st.metric("Jumlah Hari Dianalisis", f"{hari_dalam_data}")

# --- VISUALISASI TABS ---
tab_jam, tab_musim, tab_tren, tab_segmen = st.tabs([
    "Analisis Jam", 
    "Analisis Musim & Cuaca", 
    "Tren Penyewaan", 
    "Segmentasi Pelanggan"
])

# Tab Analisis Jam
with tab_jam:
    st.header("Pola Penyewaan Berdasarkan Jam")
    
    jam_df = analisis_jam(filtered_hourly)
    
    # Visualisasi dengan Plotly
    fig = px.bar(
        jam_df, 
        x='jam', 
        y='total_sewa',
        labels={'jam': 'Jam', 'total_sewa': 'Total Penyewaan'},
        title="Distribusi Penyewaan Sepeda Berdasarkan Jam"
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Insight jam tersibuk dan tersepi
    col1, col2 = st.columns(2)
    
    with col1:
        jam_sibuk = jam_df.loc[jam_df['total_sewa'].idxmax()]
        st.info(f"📈 **Jam Tersibuk**: {int(jam_sibuk['jam'])}:00 dengan {int(jam_sibuk['total_sewa']):,} penyewaan")
    
    with col2:
        jam_sepi = jam_df.loc[jam_df['total_sewa'].idxmin()]
        st.info(f"📉 **Jam Tersepi**: {int(jam_sepi['jam'])}:00 dengan {int(jam_sepi['total_sewa']):,} penyewaan")
    
    st.markdown("### Insights")
    st.write("""
    - Terdapat dua puncak penyewaan sepeda: di pagi hari dan sore hari, yang kemungkinan menunjukkan pola komuter
    - Periode tengah malam hingga pagi hari memiliki penyewaan terendah
    - Jam sibuk terjadi pada periode pergi dan pulang kantor
    """)

# Tab Analisis Musim & Cuaca
with tab_musim:
    st.header("Pola Penyewaan Berdasarkan Musim dan Cuaca")
    
    col1, col2 = st.columns(2)
    
    with col1:
        musim_df = analisis_musim(filtered_daily)
        
        fig = px.bar(
            musim_df,
            x='season',
            y='total_sewa',
            labels={'season': 'Musim', 'total_sewa': 'Total Penyewaan'},
            title="Penyewaan Berdasarkan Musim",
            text='total_sewa',
            color='season',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cuaca_df = analisis_cuaca(filtered_hourly)
        
        fig = px.bar(
            cuaca_df,
            x='kondisi_cuaca',
            y='total_sewa',
            labels={'kondisi_cuaca': 'Kondisi Cuaca', 'total_sewa': 'Total Penyewaan'},
            title="Penyewaan Berdasarkan Kondisi Cuaca",
            text='total_sewa',
            color='kondisi_cuaca',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Analisis berdasarkan kategori hari
    st.subheader("Perbedaan Pola Penyewaan pada Hari Kerja vs Akhir Pekan")
    
    kategori_hari_df = analisis_kategori_hari(filtered_daily)
    
    fig = px.bar(
        kategori_hari_df,
        x='kategori_hari',
        y=['total_sewa', 'rata_rata_sewa'],
        barmode='group',
        labels={
            'kategori_hari': 'Kategori Hari', 
            'value': 'Jumlah Penyewaan',
            'variable': 'Metrik'
        },
        title="Perbandingan Penyewaan Berdasarkan Kategori Hari"
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Analisis berdasarkan kelembaban
    st.subheader("Pengaruh Kelembaban Terhadap Penyewaan")
    
    kelembaban_df = analisis_kelembaban(filtered_hourly)
    
    fig = px.bar(
        kelembaban_df,
        x='kategori_kelembaban',
        y='rata_rata_sewa',
        labels={
            'kategori_kelembaban': 'Kategori Kelembaban', 
            'rata_rata_sewa': 'Rata-rata Penyewaan'
        },
        title="Rata-rata Penyewaan Berdasarkan Tingkat Kelembaban",
        color='kategori_kelembaban',
        color_discrete_sequence=px.colors.sequential.Blues
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tab Tren Penyewaan
with tab_tren:
    st.header("Tren Penyewaan Sepeda")
    
    tren_bulanan = analisis_tren_bulanan(filtered_daily)
    
    # Visualisasi tren bulanan
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tren_bulanan['dteday'], 
        y=tren_bulanan['total_sewa'],
        mode='lines+markers',
        name='Total Penyewaan',
        line=dict(color='royalblue', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Tren Bulanan Penyewaan Sepeda",
        xaxis_title="Bulan",
        yaxis_title="Jumlah Penyewaan",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Visualisasi tren bulanan dengan breakdown casual vs registered
    st.subheader("Tren Bulanan: Pengguna Casual vs Terdaftar")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=tren_bulanan['dteday'], 
        y=tren_bulanan['casual'],
        mode='lines+markers',
        name='Pengguna Casual',
        line=dict(color='coral', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=tren_bulanan['dteday'], 
        y=tren_bulanan['registered'],
        mode='lines+markers',
        name='Pengguna Terdaftar',
        line=dict(color='mediumseagreen', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Tren Bulanan Berdasarkan Jenis Pengguna",
        xaxis_title="Bulan",
        yaxis_title="Jumlah Penyewaan",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tabel data bulanan
    st.subheader("Data Bulanan")
    
    tren_display = tren_bulanan.copy()
    tren_display['bulan_tahun'] = tren_display['dteday'].dt.strftime('%B %Y')
    tren_display = tren_display[['bulan_tahun', 'total_sewa', 'casual', 'registered']]
    tren_display.columns = ['Bulan', 'Total Penyewaan', 'Pengguna Casual', 'Pengguna Terdaftar']
    
    st.dataframe(tren_display, use_container_width=True)

# Tab Segmentasi Pelanggan
with tab_segmen:
    st.header("Segmentasi Pelanggan")
    
    segmen_df = analisis_segmentasi_pelanggan(filtered_daily)
    
    col1, col2 = st.columns(2)
    
    # Pie chart 
    with col1:
        fig = px.pie(
            segmen_df, 
            values='jumlah', 
            names='jenis_pelanggan',
            title='Proporsi Jenis Pelanggan',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart
    with col2:
        fig = px.bar(
            segmen_df,
            x='jenis_pelanggan',
            y='jumlah',
            title='Jumlah Penyewaan per Jenis Pelanggan',
            text='jumlah',
            color='jenis_pelanggan',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=400)
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Analisis perbandingan harian
    st.subheader("Perbandingan Penyewaan Harian: Casual vs Terdaftar")
    
    daily_compare = filtered_daily.groupby('hari').agg({
        'casual': 'mean',
        'registered': 'mean'
    }).reset_index()
    
    # Mengatur ulang urutan hari
    hari_urutan = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']
    daily_compare['hari'] = pd.Categorical(daily_compare['hari'], categories=hari_urutan, ordered=True)
    daily_compare = daily_compare.sort_values('hari')
    
    daily_compare_long = pd.melt(
        daily_compare, 
        id_vars=['hari'],
        value_vars=['casual', 'registered'],
        var_name='jenis_pelanggan',
        value_name='rata_rata_sewa'
    )
    
    daily_compare_long['jenis_pelanggan'] = daily_compare_long['jenis_pelanggan'].map({
        'casual': 'Pengguna Casual',
        'registered': 'Pengguna Terdaftar'
    })
    
    fig = px.line(
        daily_compare_long,
        x='hari',
        y='rata_rata_sewa',
        color='jenis_pelanggan',
        markers=True,
        labels={
            'hari': 'Hari', 
            'rata_rata_sewa': 'Rata-rata Penyewaan',
            'jenis_pelanggan': 'Jenis Pelanggan'
        },
        title="Rata-rata Penyewaan Berdasarkan Hari dalam Seminggu"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Insights")
    st.write("""
    - Pengguna terdaftar mendominasi penyewaan sepeda, menunjukkan basis pelanggan yang loyal
    - Pengguna casual memiliki pola penggunaan yang berbeda dari pengguna terdaftar:
      - Pengguna casual cenderung lebih aktif di akhir pekan
      - Pengguna terdaftar memiliki pola yang lebih konsisten sepanjang minggu
    - Strategi pemasaran yang berbeda mungkin diperlukan untuk kedua segmen pengguna
    """)

# Tampilkan dataset asli (optional)
st.sidebar.markdown("---")
show_raw_data = st.sidebar.checkbox("Tampilkan Dataset Asli")

if show_raw_data:
    tab_harian, tab_jam = st.tabs(["Dataset Harian", "Dataset Per Jam"])
    
    with tab_harian:
        st.dataframe(data_harian)
    
    with tab_jam:
        st.dataframe(data_jam)

# Tambahkan footer
st.markdown("---")
st.markdown("Powered by Nur Fadilah Zulfi")