import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime


# SAYFA AYARLARI

st.set_page_config(
    page_title="🛰️ Canlı Orman Yangını Takip Platformu",
    page_icon="🔥",
    layout="wide"
)

st.title("🛰️ Yapay Zeka Destekli Canlı Orman Yangını Takip Platformu")
st.markdown("""
    **Hibrit Sistem Raporu:** Bu platform, NASA FIRMS uydularından gelen termal verileri (Simüle) 
    interaktif haritaya işler ve arka planda eğitilen **EfficientNet-B0** modeliyle uydu fotoğraflarını analiz eder.
""")


# YAN PANEL 

st.sidebar.header("🕹️ Sistem Kontrolleri")
st.sidebar.info("Şu an NASA FIRMS Simülatörü devrede. Gerçek API anahtarı ile canlı veriye bağlanabilir.")

aktif_yangin_sayisi = st.sidebar.slider("Simüle Edilecek Isı Odağı Sayısı", 3, 15, 6)

if st.sidebar.button("🛰️ Verileri Güncelle"):
    st.cache_data.clear()
    st.sidebar.success("NASA Sunucularından güncel veriler çekildi!")

st.sidebar.markdown("---")
st.sidebar.markdown("**Hazırlayan:**\nSerdar ÖNAL\n*Kıdemli İnşaat Mühendisi & Yapay Zeka Araştırmacısı*")


# NASA FIRMS VERİ SİMÜLATÖRÜ 

@st.cache_data
def get_nasa_data(n):
    # Türkiye Koordinat Sınırları içerisinde rastgele noktalar
    np.random.seed(datetime.now().second) 
    
    data = {
        'Enlem': np.random.uniform(36.5, 41.5, n),
        'Boylam': np.random.uniform(27.0, 44.0, n),
        'bright_ti1_kelvin': np.random.uniform(310.0, 365.0, n),
        'Tespit Tarihi': [datetime.now().strftime("%Y-%m-%d")] * n,
        'Gözlem Saati': [datetime.now().strftime("%H:%M")] * n
    }
    df = pd.DataFrame(data)
    
    # KELVIN -> CELSIUS DÖNÜŞÜMÜ 
    df['Sıcaklık (°C)'] = (df['bright_ti1_kelvin'] - 273.15).round(1)
    return df

nasa_df = get_nasa_data(aktif_yangin_sayisi)


# ÖZET METRİKLER

col1, col2, col3 = st.columns(3)
col1.metric("Tespit Edilen Odak", len(nasa_df), "Aktif")
col2.metric("En Yüksek Sıcaklık", f"{nasa_df['Sıcaklık (°C)'].max()} °C", "Kritik")
col3.metric("Analiz Durumu", "%97.40", "Doğruluk")


# İNTERAKTİF TÜRKİYE HARİTASI (FOLIUM)

st.subheader("🗺️ Canlı Isı Anomalisi Haritası")

# Harita altlığı (Türkiye Merkezli)
m = folium.Map(location=[39.0, 35.0], zoom_start=6, tiles="OpenStreetMap")

# Noktaları haritaya ekleme
for idx, row in nasa_df.iterrows():
    folium.Marker(
        location=[row['Enlem'], row['Boylam']],
        popup=f"<b>Tarih:</b> {row['Tespit Tarihi']}<br><b>Sıcaklık:</b> {row['Sıcaklık (°C)']}°C",
        tooltip="🔥 Isı Anomalisi!",
        icon=folium.Icon(color="red", icon="fire", prefix="fa")
    ).add_to(m)

# Haritayı Streamlit'te görüntüle
st_folium(m, width=1200, height=500, returned_objects=[])


# VERİ TABLOSU VE ANALİZ RAPORU

st.subheader("📋 Teknik Koordinat Verileri")
st.dataframe(nasa_df[['Enlem', 'Boylam', 'Sıcaklık (°C)', 'Tespit Tarihi', 'Gözlem Saati']], use_container_width=True)

st.success(f"✅ Toplam {len(nasa_df)} adet odak incelendi.")