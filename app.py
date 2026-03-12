import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import joblib
from rasterio.io import MemoryFile
from matplotlib.colors import ListedColormap

# تحميل النموذج المدرب
model = joblib.load("model.pkl")

# إعداد الصفحة
st.set_page_config(page_title="Raster Classification App", layout="wide")

# معلومات النموذج
st.sidebar.title("Model Information")
st.sidebar.write("Model: Decision Tree")
st.sidebar.write("Classes:")
st.sidebar.write("1 = Urban")
st.sidebar.write("2 = Agricu")
st.sidebar.write("3 = Water")

# عنوان التطبيق
st.title("Raster Classification using Decision Tree ")
st.title("محمد سليم أبو اسد ")

# رفع الصورة
uploaded_file = st.file_uploader("Upload GeoTIFF Image", type=["tif","tiff"])

if uploaded_file is not None:

    with MemoryFile(uploaded_file.read()) as memfile:
        with memfile.open() as src:

            st.success("Image loaded successfully")

            band_count = src.count
            st.write("Number of bands:", band_count)

            if band_count < 3:
                st.error("Image must contain at least 3 bands")

            else:

                band1 = st.selectbox("Band 1", list(range(1, band_count+1)),0)
                band2 = st.selectbox("Band 2", list(range(1, band_count+1)),1)
                band3 = st.selectbox("Band 3", list(range(1, band_count+1)),2)

                b1 = src.read(band1)
                b2 = src.read(band2)
                b3 = src.read(band3)

                rgb = np.dstack((b1,b2,b3))

                st.subheader("Original Image")

                fig,ax = plt.subplots()
                ax.imshow(rgb/np.max(rgb))
                ax.axis("off")
                st.pyplot(fig)

                if st.button("Run Classification"):

                    h,w,b = rgb.shape
                    X = rgb.reshape(h*w,b)

                    pred = model.predict(X)
                    classified = pred.reshape(h,w)

                    st.subheader("Classified Image")

                    cmap = ListedColormap(["gray","green","blue"])

                    fig2,ax2 = plt.subplots()
                    ax2.imshow(classified,cmap=cmap)
                    ax2.axis("off")
                    st.pyplot(fig2)

                    profile = src.profile
                    profile.update(count=1)

                    with rasterio.open("classified_output.tif","w",**profile) as dst:
                        dst.write(classified.astype(rasterio.uint8),1)

                    with open("classified_output.tif","rb") as file:
                        st.download_button("Download Result",file,"classified_output.tif")
