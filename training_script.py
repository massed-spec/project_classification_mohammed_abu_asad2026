#!/usr/bin/env python
# coding: utf-8

# #  استيراد المكتبات (Importing Required Libraries)
# ### في بداية العمل تم استيراد المكتبات اللازمة لمعالجة البيانات الفضائية وبناء نموذج التعلم الآلي.
# 

# In[43]:


import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd

from rasterio.mask import mask

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

import joblib


# #  قراءة الصورة الفضائية (Reading the Raster Image)

# ### بعد استيراد المكتبات، تم قراءة الصورة الفضائية التي تم إنشاؤها سابقا في برنامج QGIS بعد عملية دمج الباندات، وهي الملف:
# 
# RGB_stack.tif
# 
# تم استخدام مكتبة rasterio لفتح ملف الراستر وقراءة خصائصه الأساسية.

# In[57]:


raster = rasterio.open("RGB_stack.tif")

print("عدد الباندات:", raster.count)
print("عرض الصورة:", raster.width)
print("ارتفاع الصورة:", raster.height)
print("نظام الإحداثيات:", raster.crs)


# #  قراءة طبقات التدريب والتحقق

# ### (Reading Training and Validation Layers)

# ###  بعد قراءة الصورة الفضائية، تم تحميل طبقات التدريب والتحقق التي تم إنشاؤها في برنامج QGIS.
# 
# تم استخدام مكتبة geopandas لقراءة ملفات Shapefile.

# In[58]:


train_gdf = gpd.read_file("training_samples.shp")
val_gdf = gpd.read_file("validation_samples.shp")

print(train_gdf.head())
print(val_gdf.head())


# # استخراج القيم الطيفية للبكسلات
# ### (Extracting Pixel Values from Raster)

# In[59]:


def extract_samples(raster_path, shapefile_gdf):
    samples = []

    with rasterio.open(raster_path) as src:
        for _, row in shapefile_gdf.iterrows():
            geom = [row.geometry]
            class_value = row["class"]
            type_value = row["type"]

            out_image, out_transform = mask(src, geom, crop=True)

            band1 = out_image[0].flatten()
            band2 = out_image[1].flatten()
            band3 = out_image[2].flatten()

            for b1, b2, b3 in zip(band1, band2, band3):
                if b1 != 0 or b2 != 0 or b3 != 0:
                    samples.append([b1, b2, b3, class_value, type_value])

    df = pd.DataFrame(samples, columns=["band1", "band2", "band3", "class", "type"])
    return df


# ## تنفيذ الدالة واستخراج البيانات

# #### (Running the Extraction Function)
# 

# In[60]:


train_data = extract_samples("RGB_stack.tif", train_gdf)
val_data = extract_samples("RGB_stack.tif", val_gdf)

print("Training Data:")
display(train_data.head())

print("Validation Data:")
display(val_data.head())


# # عرض عدد العينات المستخرجة
# ### (Displaying the Number of Extracted Samples)

# In[61]:


print("عدد عينات التدريب:", len(train_data))
print("عدد عينات التحقق:", len(val_data))


# # تجهيز بيانات النموذج
# ## (Preparing Data for the Model)

# In[62]:


X_train = train_data[["band1","band2","band3"]]
y_train = train_data["class"]

X_val = val_data[["band1","band2","band3"]]
y_val = val_data["class"]

print("شكل بيانات التدريب:", X_train.shape)
print("شكل بيانات التحقق:", X_val.shape)


# # إنشاء النموذج
# ## (Creating the Model)

# In[63]:


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=10, random_state=42)


# # تدريب النموذج
# ### (Training the Model)

# In[65]:


model.fit(X_train, y_train)
print("تم تدريب النموذج بنجاح")


# # التنبؤ باستخدام بيانات التحقق
# ## (Prediction on Validation Data)
# 

# In[66]:


y_pred = model.predict(X_val)

print("أول 10 قيم متوقعة:")
print(y_pred[:10])


# # تقييم أداء النموذج
# ### (Model Evaluation)
# 

# ##  1. حساب الدقة
# ### (Accuracy)

# In[67]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)


# ## 2. حساب Precision و Recall

# In[68]:


precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')

print("Precision:", precision)
print("Recall:", recall)


# ## مصفوفة الالتباس
# ### (Confusion Matrix)

# In[69]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# حساب مصفوفة الالتباس
cm = confusion_matrix(y_val, y_pred)

# طباعة القيم الرقمية
print("Confusion Matrix:")
print(cm)

# أسماء الفئات
labels = ["Urban", "Agricu", "Water"]

# إنشاء الرسم
fig, ax = plt.subplots(figsize=(6,6))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap="Blues", values_format="d")

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# حفظ الصورة
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")

# عرض الصورة
plt.show()


# ## التقرير التفصيلي للتصنيف
# ### (Classification Report)

# In[71]:


from sklearn.metrics import classification_report

report = classification_report(y_val, y_pred)
print(report)


# # حفظ النموذج
# ## (Saving the Trained Model)

# In[72]:


import joblib

joblib.dump(model, "model.pkl")
print("تم حفظ النموذج   model.pkl")


# In[ ]:




