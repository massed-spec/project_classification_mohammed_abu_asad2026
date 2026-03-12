#!/usr/bin/env python
# coding: utf-8

import rasterio
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from rasterio.mask import mask
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


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


if __name__ == "__main__":
    # قراءة الصورة الفضائية
    raster = rasterio.open("RGB_stack.tif")

    print("عدد الباندات:", raster.count)
    print("عرض الصورة:", raster.width)
    print("ارتفاع الصورة:", raster.height)
    print("نظام الإحداثيات:", raster.crs)

    # قراءة طبقات التدريب والتحقق
    train_gdf = gpd.read_file("training_samples.shp")
    val_gdf = gpd.read_file("validation_samples.shp")

    print(train_gdf.head())
    print(val_gdf.head())

    # استخراج البيانات
    train_data = extract_samples("RGB_stack.tif", train_gdf)
    val_data = extract_samples("RGB_stack.tif", val_gdf)

    print("Training Data:")
    print(train_data.head())

    print("Validation Data:")
    print(val_data.head())

    # عرض عدد العينات
    print("عدد عينات التدريب:", len(train_data))
    print("عدد عينات التحقق:", len(val_data))

    # تجهيز بيانات النموذج
    X_train = train_data[["band1", "band2", "band3"]]
    y_train = train_data["class"]

    X_val = val_data[["band1", "band2", "band3"]]
    y_val = val_data["class"]

    print("شكل بيانات التدريب:", X_train.shape)
    print("شكل بيانات التحقق:", X_val.shape)

    # إنشاء النموذج
    model = DecisionTreeClassifier(max_depth=10, random_state=42)

    # تدريب النموذج
    model.fit(X_train, y_train)
    print("تم تدريب النموذج بنجاح")

    # التنبؤ
    y_pred = model.predict(X_val)

    print("أول 10 قيم متوقعة:")
    print(y_pred[:10])

    # Accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)

    # Precision و Recall
    precision = precision_score(y_val, y_pred, average="weighted")
    recall = recall_score(y_val, y_pred, average="weighted")

    print("Precision:", precision)
    print("Recall:", recall)

    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    print("Confusion Matrix:")
    print(cm)

    labels = ["Urban", "Agricu", "Water"]

    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", values_format="d")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Classification Report
    report = classification_report(y_val, y_pred)
    print(report)

    # حفظ النموذج
    joblib.dump(model, "model.pkl")
    print("تم حفظ النموذج model.pkl")
