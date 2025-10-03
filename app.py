import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import io
import seaborn as sns
from streamlit_shap import st_shap  # 需安装: pip install streamlit-shap
import os

# -------------------------
# 中文字体配置（核心修复）
# -------------------------
def set_chinese_font():
    """配置Matplotlib支持中文字体，解决方框显示问题"""
    try:
        # 优先使用系统自带中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常
    except Exception:
        # 备选方案
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Heiti TC', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False


# 初始化字体配置
set_chinese_font()

# -------------------------
# 页面配置
# -------------------------
st.set_page_config(layout="wide", page_title="脓毒症风险评估系统 — XGBoost + SHAP")

# -------------------------
# 核心配置参数
# -------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_model.json")  # 模型路径


# 特征列表（与模型训练时保持一致）
FEATURE_NAMES = [
    'temperature_avg', 'calcium_avg', 'wbc_avg', 'platelet_avg',
    'apsiii_max', 'sofa_resp_max', 'sofa_liver_max', 'sofa_cns_max',
    'sofa_cardio_max', 'sofa_renal_max'
]

PROBABILITY_FORMAT = "{:.1%}"  # 概率显示格式

# -------------------------
# 临床参考值配置
# -------------------------
RISK_THRESHOLDS = {
    "low": 0.3,  # 低风险 (<30%)
    "medium": 0.7,  # 中风险 (30%-70%)
    # 高风险 (>70%)
}

FEATURE_CRITERIA = {
    'temperature_avg': {
        'normal': (36.1, 37.2),  # 正常体温范围(°C)
        'high': 38.0,  # 发热阈值
        'critical': 39.5,  # 高热阈值
        'description': '体温'
    },
    'calcium_avg': {
        'normal': (2.1, 2.6),  # 正常血钙(mmol/L)
        'low': 2.0,  # 低钙血症阈值
        'critical': 1.75,  # 严重低钙阈值
        'description': '血钙水平'
    },
    'wbc_avg': {
        'normal': (4, 10),  # 正常白细胞(×10^9/L)
        'high': 12,  # 白细胞升高阈值
        'critical': 15,  # 显著升高阈值
        'description': '白细胞计数'
    },
    'platelet_avg': {
        'normal': (150, 450),  # 正常血小板(×10^9/L)
        'low': 100,  # 血小板减少阈值
        'critical': 50,  # 严重减少阈值
        'description': '血小板计数'
    },
    'apsiii_max': {
        'normal': (0, 40),  # APACHE III评分（越低越好）
        'high': 60,  # 高风险阈值
        'critical': 80,  # 极高风险阈值
        'description': 'APACHE III评分'
    },
    'sofa_resp_max': {
        'normal': (0, 2),  # SOFA呼吸评分
        'high': 3,  # 呼吸功能受损
        'critical': 4,  # 严重呼吸衰竭
        'description': '呼吸功能SOFA评分'
    },
    'sofa_liver_max': {
        'normal': (0, 2),  # SOFA肝脏评分
        'high': 3,  # 肝功能受损
        'critical': 4,  # 严重肝功能衰竭
        'description': '肝功能SOFA评分'
    },
    'sofa_cns_max': {
        'normal': (0, 2),  # SOFA中枢神经评分
        'high': 3,  # 意识障碍
        'critical': 4,  # 昏迷
        'description': '中枢神经SOFA评分'
    },
    'sofa_cardio_max': {
        'normal': (0, 2),  # SOFA心血管评分
        'high': 3,  # 心功能不全
        'critical': 4,  # 严重心功能衰竭
        'description': '心血管功能SOFA评分'
    },
    'sofa_renal_max': {
        'normal': (0, 2),  # SOFA肾脏评分
        'high': 3,  # 肾功能受损
        'critical': 4,  # 肾功能衰竭
        'description': '肾功能SOFA评分'
    }
}

# 系统与脓毒症关联知识
SYSTEM_SEPSIS_LINKS = {
    'respiratory': {
        'factors': ['sofa_resp_max', 'wbc_avg', 'temperature_avg'],
        'description': '呼吸系统感染（如肺炎）是脓毒症最常见的诱因',
        'prevention': [
            '加强肺部感染控制，及时使用针对性抗生素',
            '保持呼吸道通畅，必要时进行痰液培养',
            '监测血氧饱和度，维持氧合稳定'
        ]
    },
    'renal': {
        'factors': ['sofa_renal_max', 'apsiii_max'],
        'description': '泌尿系统感染或肾功能不全易进展为脓毒症',
        'prevention': [
            '监测尿量和肾功能指标（肌酐、尿素氮）',
            '避免肾毒性药物，维持充足的肾脏灌注',
            '及时处理泌尿系统感染'
        ]
    },
    'cardiovascular': {
        'factors': ['sofa_cardio_max', 'apsiii_max'],
        'description': '心血管功能障碍会加重脓毒症并降低治疗反应',
        'prevention': [
            '维持稳定的血流动力学，必要时使用血管活性药物',
            '监测乳酸水平，评估组织灌注情况',
            '避免容量过负荷'
        ]
    },
    'hematological': {
        'factors': ['platelet_avg', 'wbc_avg'],
        'description': '血液系统异常提示严重感染和炎症反应',
        'prevention': [
            '监测凝血功能，预防DIC发生',
            '必要时考虑输注血小板或血液制品',
            '评估感染源，针对性抗感染治疗'
        ]
    }
}


# -------------------------
# 辅助函数
# -------------------------
@st.cache_resource
def load_model(path):
    """加载XGBoost模型（支持booster和sklearn包装器两种格式）"""
    try:
        booster = xgb.Booster()
        booster.load_model(path)
        return {"type": "booster", "model": booster}
    except Exception:
        pass
    try:
        clf = xgb.XGBClassifier()
        clf.load_model(path)
        return {"type": "skclf", "model": clf}
    except Exception as e:
        st.error(f"模型加载失败: {e}")
        raise


def predict_proba_loaded(model_dict, X_df):
    """使用加载的模型进行概率预测"""
    if model_dict["type"] == "booster":
        d = xgb.DMatrix(X_df, feature_names=X_df.columns.tolist())
        return model_dict["model"].predict(d)
    else:
        return model_dict["model"].predict_proba(X_df)[:, 1]


def explain_single(model_dict, X_row):
    """为单个样本生成SHAP解释"""
    explainer = shap.TreeExplainer(model_dict["model"])
    shap_vals = explainer.shap_values(X_row)
    return explainer, shap_vals


def draw_shap_bar_custom(explainer, shap_values, X_row):
    """绘制美化的单个样本SHAP条形图（带中文字体支持）"""
    set_chinese_font()  # 确保字体配置

    shap_df = pd.DataFrame({
        "feature": X_row.columns,
        "shap_value": shap_values[0],
        "value": X_row.iloc[0].values
    }).sort_values("shap_value", key=abs, ascending=True)

    colors = ["tomato" if v > 0 else "steelblue" for v in shap_df["shap_value"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(
        x=shap_df["shap_value"],
        y=shap_df["feature"],
        hue=None,
        dodge=False,
        palette=colors,
        legend=False,
        ax=ax
    )

    ax.axvline(0, color="black", linestyle="--", lw=1)
    for i, shap_val in enumerate(shap_df["shap_value"]):
        ax.text(shap_val, i, f"{shap_val:.2f}", va="center",
                ha="left" if shap_val > 0 else "right")

    ax.set_title("特征对预测的影响 (SHAP值)", fontsize=14, weight="bold")
    ax.set_xlabel("SHAP值 (对模型输出的影响)", fontsize=12)
    ax.set_ylabel("特征", fontsize=12)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    return fig


def plot_shap_bar(model, X, top_n=10):
    """绘制批量特征重要性条形图（带中文字体支持）"""
    set_chinese_font()  # 确保字体配置

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=X.columns)
    mean_shap = shap_df.abs().mean().sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(8, 6))
    colors = sns.color_palette("viridis", len(mean_shap))
    sns.barplot(
        x=mean_shap.values,
        y=mean_shap.index,
        palette=colors
    )

    plt.title("特征重要性 (平均|SHAP值|)", fontsize=14, weight='bold')
    plt.xlabel("平均|SHAP值|", fontsize=12)
    plt.ylabel("特征", fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)


def get_abnormal_features(X_row):
    """识别异常指标并分类（偏高/过高）"""
    abnormal = {
        'high': [],  # 偏高
        'critical': []  # 过高/严重异常
    }

    for feature in X_row.columns:
        val = X_row.iloc[0][feature]
        criteria = FEATURE_CRITERIA[feature]
        desc = criteria['description']
        normal_low, normal_high = criteria['normal']

        # 判断高值异常
        if 'high' in criteria and val > criteria['high']:
            if val > criteria['critical']:
                abnormal['critical'].append({
                    'feature': feature,
                    'value': val,
                    'description': desc,
                    'threshold': criteria['critical'],
                    'normal_range': f"{normal_low}-{normal_high}"
                })
            elif val > normal_high:
                abnormal['high'].append({
                    'feature': feature,
                    'value': val,
                    'description': desc,
                    'threshold': criteria['high'],
                    'normal_range': f"{normal_low}-{normal_high}"
                })

        # 判断低值异常
        if 'low' in criteria and val < criteria['low']:
            if val < criteria['critical']:
                abnormal['critical'].append({
                    'feature': feature,
                    'value': val,
                    'description': desc,
                    'threshold': criteria['critical'],
                    'normal_range': f"{normal_low}-{normal_high}"
                })
            elif val < normal_low:
                abnormal['high'].append({
                    'feature': feature,
                    'value': val,
                    'description': desc,
                    'threshold': criteria['low'],
                    'normal_range': f"{normal_low}-{normal_high}"
                })

    return abnormal


def identify_primary_risk_system(shap_values, X_row):
    """识别最可能的脓毒症诱导系统"""
    shap_df = pd.DataFrame({
        "feature": X_row.columns,
        "shap_value": shap_values[0],
        "value": X_row.iloc[0].values
    }).sort_values("shap_value", ascending=False)

    # 计算各系统的风险分数
    system_scores = {system: 0 for system in SYSTEM_SEPSIS_LINKS}

    for system, info in SYSTEM_SEPSIS_LINKS.items():
        for feature in info['factors']:
            shap_val = shap_df[shap_df['feature'] == feature]['shap_value'].values[0]
            if shap_val > 0:  # 只计算增加风险的特征
                system_scores[system] += abs(shap_val)

    # 找到风险分数最高的系统
    primary_system = max(system_scores, key=system_scores.get)
    return primary_system, SYSTEM_SEPSIS_LINKS[primary_system]


def generate_recommendations(risk_prob, X_row, shap_values):
    """生成预后建议"""
    recommendations = []
    abnormal = get_abnormal_features(X_row)

    # 1. 风险等级评估
    if risk_prob < RISK_THRESHOLDS["low"]:
        risk_level = "低"
        recommendations.append(
            f"患者脓毒症风险评估为{risk_level}（{PROBABILITY_FORMAT.format(risk_prob)}），总体预后良好。")
    elif risk_prob < RISK_THRESHOLDS["medium"]:
        risk_level = "中"
        recommendations.append(
            f"患者脓毒症风险评估为{risk_level}（{PROBABILITY_FORMAT.format(risk_prob)}），需要密切观察病情变化。")
    else:
        risk_level = "高"
        recommendations.append(
            f"患者脓毒症风险评估为{risk_level}（{PROBABILITY_FORMAT.format(risk_prob)}），建议立即采取干预措施。")

    # 2. 异常指标分析
    if abnormal['critical'] or abnormal['high']:
        recommendations.append("\n**异常指标分析：**")

        if abnormal['critical']:
            recommendations.append("  - 严重异常指标（需紧急处理）：")
            for item in abnormal['critical']:
                recommendations.append(
                    f"    - {item['description']}：{item['value']:.1f}（正常范围：{item['normal_range']}），已显著超出警戒值{item['threshold']}")

        if abnormal['high']:
            recommendations.append("  - 偏高指标（需密切监测）：")
            for item in abnormal['high']:
                recommendations.append(
                    f"    - {item['description']}：{item['value']:.1f}（正常范围：{item['normal_range']}），超出警戒值{item['threshold']}")
    else:
        recommendations.append("\n**指标分析：** 所有监测指标均在正常范围内。")

    # 3. 最可能的诱导因素
    primary_system, system_info = identify_primary_risk_system(shap_values, X_row)
    system_names = {
        'respiratory': '呼吸系统',
        'renal': '肾脏/泌尿系统',
        'cardiovascular': '心血管系统',
        'hematological': '血液系统'
    }
    recommendations.append(f"\n**最可能的脓毒症诱导因素：** {system_names[primary_system]}")
    recommendations.append(f"  - 原因分析：{system_info['description']}")

    # 4. 预防与干预建议
    recommendations.append("\n**预防与干预建议：**")
    for measure in system_info['prevention']:
        recommendations.append(f"  - {measure}")

    # 5. 通用监测计划
    recommendations.append("\n**通用监测计划：**")
    if risk_level == "高":
        recommendations.append("  - 每4-6小时监测生命体征和关键指标")
        recommendations.append("  - 考虑经验性广谱抗生素治疗")
        recommendations.append("  - 评估感染源，必要时进行微生物培养")
        recommendations.append("  - 维持足够的组织灌注和氧供")
    elif risk_level == "中":
        recommendations.append("  - 每12小时监测生命体征和关键指标")
        recommendations.append("  - 完善感染筛查，密切观察病情变化")
        recommendations.append("  - 优化液体管理，维持内环境稳定")
    else:
        recommendations.append("  - 每日监测生命体征和关键指标")
        recommendations.append("  - 继续原发病治疗，预防感染风险因素")
        recommendations.append("  - 如出现发热、白细胞升高等症状立即复查")

    return "\n".join(recommendations)


# -------------------------
# 页面UI
# -------------------------
st.title("脓毒症风险评估系统 — XGBoost + SHAP")
st.write("输入患者24小时内的临床指标，系统将评估脓毒症风险并提供临床建议。")

# 单患者输入
with st.expander("患者信息输入", True):
    cols = st.columns(3)
    input_vals = {}
    for i, feat in enumerate(FEATURE_NAMES):
        col = cols[i % 3]
        criteria = FEATURE_CRITERIA[feat]
        normal_low, normal_high = criteria['normal']

        # 根据特征类型设置合理的输入范围和提示
        if "temperature" in feat:
            val = col.number_input(
                f"{criteria['description']} ({feat})",
                value=36.5,
                min_value=34.0,
                max_value=42.0,
                step=0.1,
                help=f"正常范围: {normal_low}-{normal_high}°C"
            )
        elif "wbc" in feat:
            val = col.number_input(
                f"{criteria['description']} ({feat})",
                value=7.0,
                min_value=0.0,
                max_value=50.0,
                step=0.1,
                help=f"正常范围: {normal_low}-{normal_high}×10^9/L"
            )
        elif "calcium" in feat:
            val = col.number_input(
                f"{criteria['description']} ({feat})",
                value=2.3,
                min_value=0.0,
                max_value=5.0,
                step=0.01,
                help=f"正常范围: {normal_low}-{normal_high}mmol/L"
            )
        elif "platelet" in feat:
            val = col.number_input(
                f"{criteria['description']} ({feat})",
                value=300.0,
                min_value=0.0,
                max_value=1000.0,
                step=10.0,
                help=f"正常范围: {normal_low}-{normal_high}×10^9/L"
            )
        else:  # SOFA评分和APACHE评分
            val = col.number_input(
                f"{criteria['description']} ({feat})",
                value=0.0,
                min_value=0.0,
                max_value=20.0 if 'apsiii' in feat else 4.0,
                step=1.0,
                help=f"正常范围: {normal_low}-{normal_high}"
            )
        input_vals[feat] = val
    analyze_btn = st.button("分析患者风险")

# 批量上传
with st.expander("批量分析（CSV上传）", False):
    uploaded = st.file_uploader("上传CSV文件", type=["csv"])
    if uploaded is not None:
        try:
            df_uploaded = pd.read_csv(uploaded)
            st.success(f"已加载 {len(df_uploaded)} 条记录。预览：")
            st.dataframe(df_uploaded.head())
        except Exception as e:
            st.error(f"CSV读取失败: {e}")
            df_uploaded = None
    else:
        df_uploaded = None

# 加载模型
model_dict = load_model(MODEL_PATH)

# -------------------------
# 单患者分析结果展示
# -------------------------
if analyze_btn:
    X_single = pd.DataFrame([input_vals], columns=FEATURE_NAMES)
    prob = predict_proba_loaded(model_dict, X_single)[0]

    # 显示风险评估结果
    st.metric("脓毒症预测风险", PROBABILITY_FORMAT.format(prob))
    st.write(f"原始概率值: {prob:.6f}")

    with st.spinner("正在计算SHAP解释并生成临床建议..."):
        explainer, shap_vals = explain_single(model_dict, X_single)

        # 显示SHAP条形图
        fig = draw_shap_bar_custom(explainer, shap_vals, X_single)
        st.pyplot(fig)

        # 交互式力图
        st.subheader("交互式影响图")
        st_shap(shap.force_plot(
            explainer.expected_value,
            shap_vals[0, :],
            X_single.iloc[0, :],
            matplotlib=False
        ))

        # 生成并显示预后建议
        st.subheader("临床预后与预防建议")
        recommendations = generate_recommendations(prob, X_single, shap_vals)
        st.info(recommendations)

    # 导出结果
    if st.button("导出该患者预测结果"):
        outdf = X_single.copy()
        outdf["pred_prob"] = prob
        buf = io.BytesIO()
        outdf.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("下载CSV结果", data=buf,
                           file_name="sepsis_prediction_single.csv", mime="text/csv")

# -------------------------
# 批量分析结果展示
# -------------------------
if df_uploaded is not None:
    missing = [c for c in FEATURE_NAMES if c not in df_uploaded.columns]
    if missing:
        st.error(f"缺少必要特征: {missing}")
    else:
        X_batch = df_uploaded[FEATURE_NAMES].copy()
        with st.spinner("正在批量预测..."):
            probs = predict_proba_loaded(model_dict, X_batch)
            df_uploaded["pred_prob"] = probs
        st.success("预测完成。预览：")
        st.dataframe(df_uploaded.head())

        # 批量特征重要性
        if st.button("显示批量特征重要性"):
            with st.spinner("正在计算批量SHAP值..."):
                plot_shap_bar(model_dict["model"], X_batch)

        # 为前N行计算详细SHAP和建议
        n_show = st.number_input("需要详细分析的记录数（前N行）", min_value=1, max_value=20, value=3)
        if st.button("计算前N行的详细分析"):
            with st.spinner("正在计算详细分析..."):
                for i in range(min(n_show, len(X_batch))):
                    st.write(f"### 记录 {i + 1}（预测概率={df_uploaded.loc[df_uploaded.index[i], 'pred_prob']:.3f}）")
                    explainer, shap_vals = explain_single(model_dict, X_batch.iloc[[i]])

                    # 显示SHAP条形图
                    fig = draw_shap_bar_custom(explainer, shap_vals, X_batch.iloc[[i]])
                    st.pyplot(fig)

                    # 显示临床建议
                    st.text("临床建议：")
                    recommendations = generate_recommendations(
                        df_uploaded.loc[df_uploaded.index[i], 'pred_prob'],
                        X_batch.iloc[[i]],
                        shap_vals
                    )
                    st.info(recommendations)
                    st.divider()

        # 下载批量结果
        buf = io.BytesIO()
        df_uploaded.to_csv(buf, index=False)
        buf.seek(0)
        st.download_button("下载批量预测结果", data=buf,
                           file_name="sepsis_predictions_batch.csv", mime="text/csv")

# 页脚
st.markdown("---")
st.caption("注：本系统提供的建议仅供临床参考，最终诊疗决策请结合患者具体情况由医疗团队制定。")
