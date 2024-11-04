import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 加载标准器和模型
scaler = joblib.load('scaler_standard_C.pkl')
model = joblib.load('lightgbm_C.pkl')

# 定义特征名称
features_to_scale = [
    'CI_age', 'CI_endometrial thickness', 'CI_HE4', 'CI_menopause',
    'CI_HRT', 'CI_endometrial heterogeneity',
    'CI_uterine cavity occupation',
    'CI_uterine cavity occupying lesion with rich blood flow',
    'CI_uterine cavity fluid'
]

additional_features = [
    'CM5141.0', 'CM6160.0', 'CM7441.0', 'CM7439.0', 
    'CM7438.0', 'CM5139.0', 'CM6557.0', 'CM4088.0'
]

all_features = features_to_scale + additional_features

# Streamlit界面
st.title("疾病风险预测器")

# 获取用户输入
user_input = {}
for feature in features_to_scale:
    user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, value=0.0)

for feature in additional_features:
    user_input[feature] = st.number_input(f"{feature}:", min_value=0.0, value=0.0)

# 提取并标准化特征
input_df = pd.DataFrame([user_input])
input_df_scaled = input_df.copy()
input_df_scaled[features_to_scale] = scaler.transform(input_df[features_to_scale])

if st.button("预测"):
    # 使用模型进行预测
    predicted_proba = model.predict_proba(input_df_scaled)[0]
    predicted_class = model.predict(input_df_scaled)[0]

    # 显示预测概率
    st.write("**预测概率:**")
    st.write(f"类别 0（无疾病）: {predicted_proba[0]:.2f}")
    st.write(f"类别 1（有疾病）: {predicted_proba[1]:.2f}")

    # 根据预测类别给出建议
    if predicted_class == 1:
        st.write(
            "**结果: 您有较高的患病风险。** 根据模型的预测结果，建议您住院接受进一步的专业医疗评估。"
        )
    else:
        st.write(
            "**结果: 您的患病风险较低。** 建议您定期进行健康检查，以便随时监控您的健康状况。"
        )
