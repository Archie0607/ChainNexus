import hashlib
import importlib

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from utils.model_loader import ARTIFACTS, load_bundle, load_metrics, load_model
from utils import evaluation as evaluation_utils
from utils import prediction as prediction_utils
from utils import preprocessing as preprocessing_utils
from utils import visualization as visualization_utils

# Streamlit can rerun app.py while retaining imported helper modules in sys.modules.
# Reload project utilities so changed helpers are available without a stale import error.
importlib.reload(preprocessing_utils)
importlib.reload(prediction_utils)
importlib.reload(visualization_utils)
importlib.reload(evaluation_utils)

display_high_threshold = prediction_utils.display_high_threshold
predict_dataframe = prediction_utils.predict_dataframe
evaluate_labeled_dataframe = evaluation_utils.evaluate_labeled_dataframe
TARGET_COLUMN = evaluation_utils.TARGET_COLUMN
validate_columns = preprocessing_utils.validate_columns
grouped_risk = visualization_utils.grouped_risk
risk_pie = visualization_utils.risk_pie

st.set_page_config(
    page_title="ChainNexus AI | ระบบประเมินความเสี่ยงจัดส่ง",
    page_icon=":material/hub:",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    bundle = load_bundle()
    metrics = load_metrics()
except Exception as exc:
    st.error(f"โหลดไฟล์โมเดลไม่สำเร็จ: {exc}", title="เปิดระบบไม่ได้")
    st.stop()

PAGES = ["ภาพรวม", "ทำนายความเสี่ยง", "ผลโมเดล", "วิธีใช้"]
FIELD_LABELS = {
    "Type": "ประเภทการชำระเงิน",
    "Days for shipment (scheduled)": "กำหนดเวลาจัดส่ง (วัน)",
    "Category Name": "หมวดสินค้า",
    "Customer Segment": "กลุ่มลูกค้า",
    "Department Name": "แผนก",
    "Market": "ตลาด",
    "Order Region": "ภูมิภาคจัดส่ง",
    "Order Item Discount Rate": "อัตราส่วนลด (เช่น 0.10 = 10%)",
    "Order Item Quantity": "จำนวนสินค้า",
    "Order Item Product Price": "ราคาสินค้าต่อชิ้น",
    "Shipping Mode": "วิธีจัดส่ง",
}
if "page_nav" not in st.session_state:
    st.session_state["page_nav"] = PAGES[0]

with st.sidebar:
    st.markdown("## :material/hub: ChainNexus")
    st.caption("ระบบช่วยประเมินความเสี่ยงจัดส่ง")
    st.divider()
    st.caption("โมเดลที่ใช้")
    st.markdown(f"**{metrics.get('selected_model', '—')}**")
    saved_threshold = metrics.get("selected_threshold", 0.34)
    high_level_threshold = display_high_threshold(saved_threshold)
    st.caption(f"เกณฑ์เตือน {saved_threshold:.0%} · ระดับสูงตั้งแต่ {high_level_threshold:.0%}")
    st.caption("ช่วง 3 ระดับเป็นเกณฑ์แสดงผลชั่วคราว")


def predict_for_analytics(source: pd.DataFrame) -> pd.DataFrame:
    cache_key = "analytics_result"
    file_key = "analytics_file_name"
    current_name = st.session_state.get("analytics_name")
    if cache_key not in st.session_state or st.session_state.get(file_key) != current_name:
        result = predict_dataframe(source, load_model(), bundle, metrics["selected_threshold"])
        st.session_state[cache_key] = result
        st.session_state[file_key] = current_name
    return st.session_state[cache_key]


def render_kpis(df: pd.DataFrame) -> None:
    total = len(df)
    high = int(df["Risk Level"].eq("สูง").sum())
    medium = int(df["Risk Level"].eq("กลาง").sum())
    low = int(df["Risk Level"].eq("ต่ำ").sum())
    rate = float(df["Risk Classification"].mean()) if total else 0
    with st.container(horizontal=True):
        st.metric("รายการที่ประเมิน", f"{total:,}", border=True)
        st.metric("เสี่ยงสูง", f"{high:,}", border=True)
        st.metric("เสี่ยงกลาง", f"{medium:,}", border=True)
        st.metric("เสี่ยงต่ำ", f"{low:,}", border=True)
        st.metric("โมเดลเตือนตามเกณฑ์", f"{rate:.1%}", border=True)


def render_priority_queue(df: pd.DataFrame, key: str) -> None:
    st.markdown("#### คิวรายการที่ควรตรวจสอบ")
    st.caption("เรียงจากโอกาสเสี่ยงส่งช้ามากไปน้อย เพื่อช่วยจัดลำดับการติดตาม")
    order_id = next(
        (c for c in df.columns if c.casefold() in {"order id", "order_id", "orderid", "order item id", "order item id"}),
        None,
    )
    display = df.copy()
    display.insert(0, "หมายเลขรายการ", display[order_id].astype(str) if order_id else [f"แถว {i+1}" for i in range(len(display))])
    columns = ["หมายเลขรายการ", "Risk Level", "Late Risk Probability", "Risk Classification"]
    columns.extend(c for c in ["Shipping Mode", "Market", "Department Name", "Customer Segment"] if c in display.columns)
    display = display.sort_values("Late Risk Probability", ascending=False).loc[:, columns]
    risk_filter = st.selectbox("กรองระดับความเสี่ยง", ["ทั้งหมด", "สูง", "กลาง", "ต่ำ"], key=f"risk_filter_{key}")
    if risk_filter != "ทั้งหมด":
        display = display[display["Risk Level"] == risk_filter]
    display = display.rename(columns={
        "หมายเลขรายการ": "รายการ/Order ID",
        "Risk Level": "ระดับความเสี่ยง",
        "Late Risk Probability": "โอกาสเสี่ยงส่งช้า",
        "Risk Classification": "เกินเกณฑ์เตือน (0/1)",
        "Shipping Mode": "วิธีจัดส่ง",
        "Market": "ตลาด",
        "Department Name": "แผนก",
        "Customer Segment": "กลุ่มลูกค้า",
    })
    display["โอกาสเสี่ยงส่งช้า"] = (display["โอกาสเสี่ยงส่งช้า"] * 100).round(1).astype(str) + "%"
    display["เกินเกณฑ์เตือน (0/1)"] = display["เกินเกณฑ์เตือน (0/1)"].map({1: "เตือน", 0: "ยังไม่เตือน"})
    st.dataframe(display, width="stretch", hide_index=True)
    st.caption("รายการ/Order ID มาจากไฟล์ของคุณ หากไฟล์ไม่มี Order ID จะแสดงหมายเลขแถวแทน")


def render_charts(df: pd.DataFrame) -> None:
    def group_summary(column: str) -> str:
        grouped = df.groupby(column, dropna=False).agg(
            Orders=("Risk Classification", "size"),
            High=("Risk Classification", "sum"),
        )
        grouped["Rate"] = grouped["High"] / grouped["Orders"]
        if len(grouped) < 2:
            label = str(grouped.index[0]) if len(grouped) else "ไม่มีข้อมูล"
            return f"ไฟล์นี้มีข้อมูลในกลุ่ม {label} เพียงกลุ่มเดียว จึงยังเปรียบเทียบกับกลุ่มอื่นไม่ได้"
        top = grouped.sort_values(["Rate", "Orders"], ascending=[False, False]).iloc[0]
        bottom = grouped.sort_values(["Rate", "Orders"], ascending=[True, False]).iloc[0]
        top_name = str(grouped.sort_values(["Rate", "Orders"], ascending=[False, False]).index[0])
        bottom_name = str(grouped.sort_values(["Rate", "Orders"], ascending=[True, False]).index[0])
        gap = (top["Rate"] - bottom["Rate"]) * 100
        return (
            f"อัตราเสี่ยงสูงสุดอยู่ที่ **{top_name}** ({top['Rate']:.0%}, {int(top['Orders'])} รายการ) "
            f"ส่วนต่ำสุดคือ **{bottom_name}** ({bottom['Rate']:.0%}, {int(bottom['Orders'])} รายการ) "
            f"ต่างกัน {gap:.0f} จุดเปอร์เซ็นต์ "
            "ใช้เป็นสัญญาณช่วยเลือกกลุ่มที่ควรตรวจสอบก่อน ไม่ได้ยืนยันว่ากลุ่มนั้นเป็นสาเหตุให้ส่งช้า"
        )

    def group_panel(container, column: str, title: str) -> None:
        with container, st.container(border=True):
            st.markdown(f"#### {title}")
            st.caption(f"แท่งแสดงสัดส่วนรายการที่โมเดลเตือนว่ามีความเสี่ยง (เกณฑ์ {metrics['selected_threshold']:.0%})")
            groups = df[column].fillna("ไม่ระบุ").astype(str).unique()
            if len(groups) == 1:
                rate = float(df["Risk Classification"].mean()) if len(df) else 0.0
                with st.container(horizontal=True):
                    st.metric("ประเภทที่พบ", groups[0], border=True)
                    st.metric("เสี่ยงสูงในกลุ่มนี้", f"{rate:.1%}", border=True)
                st.info(group_summary(column), title="สรุป")
            else:
                st.plotly_chart(
                    grouped_risk(df, column),
                    width="stretch",
                    config={"displayModeBar": False},
                    alt=title,
                )
                st.info(group_summary(column), title="สรุปและการนำไปใช้")

    st.subheader("1. ภาพรวมระดับความเสี่ยง")
    st.caption("วงแหวนแสดงสัดส่วนรายการที่โมเดลจัดเป็นเสี่ยงสูงและเสี่ยงต่ำจาก CSV นี้")
    chart_col, summary_col = st.columns([1, 1.2])
    with chart_col, st.container(border=True):
        st.plotly_chart(risk_pie(df), width="stretch", config={"displayModeBar": False}, alt="สัดส่วนรายการที่โมเดลประเมินว่าเสี่ยงสูงและเสี่ยงต่ำ")
    high = int(df["Risk Level"].eq("สูง").sum())
    medium = int(df["Risk Level"].eq("กลาง").sum())
    low = int(df["Risk Level"].eq("ต่ำ").sum())
    rate = float(df["Risk Classification"].mean()) if len(df) else 0
    with summary_col, st.container(border=True):
        st.markdown("#### สรุปภาพรวม")
        st.write(f"จากทั้งหมด **{len(df):,} รายการ**: เสี่ยงสูง {high:,} · กลาง {medium:,} · ต่ำ {low:,}")
        st.write(f"โมเดลส่งสัญญาณเตือน {high + medium:,} รายการ ({rate:.1%}) ตาม threshold ที่บันทึกไว้ ({metrics['selected_threshold']:.0%})")
        st.write("**นำไปใช้:** เริ่มตรวจรายการเสี่ยงสูงในคิวด้านล่างก่อน แล้วตรวจสอบกำหนดส่งและสถานะขนส่งในระบบปฏิบัติงานของคุณ")
        st.caption("ระดับ 3 สีเป็นการจัดกลุ่มคะแนนเพื่อช่วยอ่านผล ไม่ใช่คลาสที่โมเดลแยกเรียนรู้โดยตรง")

    st.divider()
    st.subheader("2. เปรียบเทียบความเสี่ยงระหว่างกลุ่ม")
    st.caption("แต่ละกราฟคำนวณจากสัดส่วนรายการที่โมเดลจัดเป็นเสี่ยงสูงภายในกลุ่มนั้น ๆ")
    left, right = st.columns(2)
    group_panel(left, "Shipping Mode", "วิธีจัดส่ง")
    group_panel(right, "Market", "ตลาด")
    left, right = st.columns(2)
    group_panel(left, "Department Name", "แผนกสินค้า")
    group_panel(right, "Customer Segment", "กลุ่มลูกค้า")


def prepare_uploaded_csv(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        raw = pd.read_csv(uploaded_file)
        validate_columns(raw, bundle["features"])
        fingerprint = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
        st.session_state["analytics_name"] = f"{uploaded_file.name}:{fingerprint}"
        return raw
    except Exception as exc:
        st.error(f"อ่านไฟล์ไม่สำเร็จ: {exc}", title="ตรวจสอบไฟล์ CSV")
        return None


def show_missing_input_note(df: pd.DataFrame) -> None:
    missing_numeric = int(df[bundle["numerical_features"]].isna().sum().sum())
    missing_categorical = int(df[bundle["categorical_features"]].isna().sum().sum())
    if missing_numeric or missing_categorical:
        st.warning(
            f"พบช่องว่าง: ตัวเลข {missing_numeric:,} ช่องจะเติมด้วยค่ากลางจากตอนฝึก; "
            f"หมวดหมู่ {missing_categorical:,} ช่องจะเข้ารหัสเป็นค่าไม่ทราบ",
            title="มีข้อมูลไม่ครบ",
        )


st.title(":material/hub: ChainNexus AI")
st.caption("ประเมินความเสี่ยงการจัดส่งจากข้อมูลคำสั่งซื้อ")
page = st.segmented_control(
    "เมนูหลัก",
    PAGES,
    key="page_nav",
    label_visibility="collapsed",
    width="stretch",
)
page = page or PAGES[0]
st.divider()

if page == "ภาพรวม":
    st.subheader("ภาพรวมคำสั่งซื้อ")
    st.write("อัปโหลดไฟล์คำสั่งซื้อเพื่อประเมินความเสี่ยง หรือไปที่ **ทำนายความเสี่ยง** เพื่อกรอกข้อมูลทีละรายการ")
    uploaded = st.file_uploader(
        "เลือกไฟล์ CSV คำสั่งซื้อ",
        type=["csv"],
        key="dashboard_csv",
        help="ต้องมีคอลัมน์ตามไฟล์ตัวอย่างที่ดาวน์โหลดได้จากหน้าทำนายความเสี่ยง",
    )
    raw = prepare_uploaded_csv(uploaded)
    if raw is None:
        st.download_button(
            "ดาวน์โหลดไฟล์ตัวอย่าง",
            (ARTIFACTS / "batch_prediction_template.csv").read_bytes(),
            "chainnexus_template.csv",
            "text/csv",
            type="secondary",
        )
        st.info("ยังไม่มีไฟล์ข้อมูล กรุณาอัปโหลด CSV หรือเลือกเมนู **ทำนายความเสี่ยง** เพื่อทดลองกรอกข้อมูล", title="เริ่มใช้งาน")
        with st.container(border=True):
            st.markdown("**ระบบนี้ใช้ข้อมูลอะไรบ้าง?**")
            st.write("โมเดลต้องใช้ข้อมูลคำสั่งซื้อ 11 ช่อง เช่น วิธีจัดส่ง ตลาด หมวดสินค้า จำนวนและราคา โดยไม่ใช้ผลการจัดส่งจริงมาทำนาย")
    else:
        try:
            show_missing_input_note(raw)
            scored = predict_for_analytics(raw)
            render_kpis(scored)
            st.caption(f"กำลังแสดงผลทำนายสำหรับไฟล์ **{uploaded.name}** · ผลนี้เป็นการประเมินจากโมเดล ไม่ใช่ผลจัดส่งจริง")
            render_priority_queue(scored, "dashboard")
            render_charts(scored)
        except Exception as exc:
            st.error(f"ประเมินไฟล์ไม่สำเร็จ: {exc}", title="ทำนายไม่ได้")

elif page == "ทำนายความเสี่ยง":
    st.subheader("ทำนายความเสี่ยง")
    st.write("เลือกวิธีที่สะดวก: กรอกหนึ่งรายการ หรืออัปโหลดไฟล์เพื่อทำนายหลายรายการ")
    one_tab, csv_tab = st.tabs(["กรอกหนึ่งรายการ", "อัปโหลด CSV หลายรายการ"])

    with one_tab:
        st.caption("กรอกข้อมูลคำสั่งซื้อ แล้วกดปุ่มด้านล่างเพื่อดูความเสี่ยง")
        maps = bundle["category_maps"]
        values = {}
        with st.form("single_prediction"):
            left, right = st.columns(2)
            for index, column in enumerate(bundle["features"]):
                area = left if index % 2 == 0 else right
                if column in bundle["categorical_features"]:
                    values[column] = area.selectbox(FIELD_LABELS.get(column, column), list(maps[column]), key=f"single_{column}")
                else:
                    median = float(bundle["num_imputer"].statistics_[bundle["numerical_features"].index(column)])
                    options = {"value": median, "format": "%.4f", "key": f"single_{column}"}
                    if column == "Order Item Quantity":
                        options.update(min_value=1.0, step=1.0, format="%.0f")
                    elif column == "Order Item Discount Rate":
                        options.update(min_value=0.0, max_value=1.0, step=0.01)
                    elif column in {"Days for shipment (scheduled)", "Order Item Product Price"}:
                        options.update(min_value=0.0)
                    values[column] = area.number_input(FIELD_LABELS.get(column, column), **options)
            submitted = st.form_submit_button("ประเมินความเสี่ยง", type="primary", width="stretch")
        if submitted:
            try:
                result = predict_dataframe(pd.DataFrame([values]), load_model(), bundle, metrics["selected_threshold"])
                probability = float(result.iloc[0]["Late Risk Probability"])
                high = int(result.iloc[0]["Risk Classification"])
                risk_level = result.iloc[0]["Risk Level"]
                st.divider()
                st.subheader("ผลการประเมิน")
                with st.container(horizontal=True):
                    st.metric("โอกาสเสี่ยงจัดส่งล่าช้า", f"{probability:.1%}", border=True)
                    st.metric("ระดับความเสี่ยง", risk_level, border=True)
                    st.metric("ผลตาม threshold โมเดล", "เตือน" if high else "ยังไม่เตือน", border=True)
                st.progress(min(max(probability, 0), 1), text=f"โมเดลส่งสัญญาณตั้งแต่ {metrics['selected_threshold']:.0%}")
                st.caption(f"ระดับคะแนน: ต่ำ < {metrics['selected_threshold']:.0%} · กลางถึง {display_high_threshold(metrics['selected_threshold']):.0%} · สูงตั้งแต่ {display_high_threshold(metrics['selected_threshold']):.0%}. คลาสเตือนของโมเดลยังใช้ threshold {metrics['selected_threshold']:.0%}")
                st.info("ใช้ผลเพื่อจัดลำดับการตรวจสอบคำสั่งซื้อ ไม่ใช่การยืนยันว่าจะจัดส่งล่าช้าหรือบอกสาเหตุของความล่าช้า", title="การนำผลไปใช้")
            except Exception as exc:
                st.error(f"ประเมินไม่สำเร็จ: {exc}", title="ทำนายไม่ได้")

    with csv_tab:
        st.write("เลือกไฟล์ CSV ที่มีข้อมูลคำสั่งซื้อ ระบบจะแสดงผลและให้ดาวน์โหลดได้")
        st.download_button(
            "ดาวน์โหลดไฟล์ตัวอย่าง",
            (ARTIFACTS / "batch_prediction_template.csv").read_bytes(),
            "chainnexus_template.csv",
            "text/csv",
        )
        uploaded = st.file_uploader("เลือกไฟล์ CSV", type=["csv"], key="batch_csv")
        raw = prepare_uploaded_csv(uploaded)
        if raw is not None:
            try:
                validate_columns(raw, bundle["features"])
                show_missing_input_note(raw)
                st.caption(f"พบข้อมูล {len(raw):,} รายการ")
                if st.button("ประเมินทุกรายการ", type="primary"):
                    batch_fingerprint = st.session_state.get("analytics_name")
                    st.session_state["batch_prediction"] = predict_dataframe(raw, load_model(), bundle, metrics["selected_threshold"])
                    st.session_state["batch_prediction_file"] = batch_fingerprint
                result = st.session_state.get("batch_prediction")
                if st.session_state.get("batch_prediction_file") != st.session_state.get("analytics_name"):
                    result = None
                if result is not None:
                    render_kpis(result)
                    render_priority_queue(result, "batch")
                    with st.expander("ดูข้อมูลทุกคอลัมน์จากไฟล์"):
                        st.dataframe(result, width="stretch", hide_index=True)
                    st.download_button(
                        "ดาวน์โหลดผลการประเมิน CSV",
                        result.to_csv(index=False).encode("utf-8-sig"),
                        "chainnexus_predictions.csv",
                        "text/csv",
                        type="primary",
                    )
            except Exception as exc:
                st.error(f"ตรวจสอบหรือทำนายไฟล์ไม่สำเร็จ: {exc}", title="ตรวจสอบไฟล์ CSV")

elif page == "ผลโมเดล":
    st.subheader("ผลการทดสอบโมเดล")
    st.caption("ผลจากชุดทดสอบที่ส่งออกมาพร้อมโมเดล และเครื่องมือประเมินข้อมูลที่มีผลจริงกำกับ")
    rows = metrics.get("test_metrics", [])
    selected = next((row for row in rows if row.get("Model") == metrics.get("selected_model")), None)
    if selected:
        items = [
            ("Accuracy", "ความถูกต้อง"), ("Precision", "Precision"), ("Recall", "Recall"),
            ("F1", "F1-score"), ("ROC_AUC", "ROC-AUC"),
        ]
        with st.container(horizontal=True):
            for key, label in items:
                st.metric(label, f"{selected[key]:.3f}", border=True)
        st.caption(f"Threshold {selected['Threshold']:.2f} · test set {metrics.get('test_rows', '—'):,} รายการ")
        passed_accuracy = selected["Accuracy"] >= 0.80
        passed_recall = selected["Recall"] >= 0.80
        st.warning(
            f"เป้าหมาย Accuracy ≥ 80%: {'ผ่าน' if passed_accuracy else 'ยังไม่ผ่าน'} · "
            f"Recall ≥ 80%: {'ผ่าน' if passed_recall else 'ยังไม่ผ่าน'}",
            title="เทียบกับเป้าหมายโครงงาน",
        )
    st.markdown("#### เปรียบเทียบโมเดล")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    feature_file = ARTIFACTS / "metrics" / "feature_importance.csv"
    if feature_file.exists():
        feature_data = pd.read_csv(feature_file)
        st.markdown("#### ปัจจัยที่มีผลต่อคะแนนโมเดล")
        chart = px.bar(feature_data.sort_values("Importance Mean"), x="Importance Mean", y="Feature", orientation="h")
        st.plotly_chart(chart, width="stretch", config={"displayModeBar": False}, alt="กราฟค่าความสำคัญของฟีเจอร์ที่ส่งออกมาพร้อมโมเดล")
    with st.expander("รายละเอียด Hyperparameter tuning"):
        st.json(metrics.get("hyperparameter_tuning", {}), expanded=False)

    st.divider()
    st.markdown("#### ประเมินกับข้อมูลที่มีผลจัดส่งจริง")
    st.write(f"อัปโหลด CSV ที่มี 11 ฟีเจอร์ของโมเดลและคอลัมน์ `{TARGET_COLUMN}` (0 = ไม่ล่าช้า, 1 = ล่าช้า) เพื่อสร้าง Confusion Matrix, ROC และ Precision–Recall จากข้อมูลนั้น")
    st.warning("ใช้เฉพาะชุดข้อมูลที่ไม่ได้นำไปฝึกหรือเลือก threshold มาก่อน หากใช้ข้อมูลฝึกเดิม ผลจะมองโลกดีเกินจริง", title="เลือกข้อมูลประเมิน")
    eval_file = st.file_uploader("CSV สำหรับประเมิน", type=["csv"], key="evaluation_csv")
    if eval_file is not None:
        eval_hash = hashlib.sha256(eval_file.getvalue()).hexdigest()
        try:
            eval_data = pd.read_csv(eval_file)
            validate_columns(eval_data, bundle["features"])
            if TARGET_COLUMN not in eval_data.columns:
                st.error(f"ไม่พบคอลัมน์ `{TARGET_COLUMN}` ในไฟล์")
            else:
                st.caption(f"พบข้อมูลประเมิน {len(eval_data):,} รายการ")
                if st.button("คำนวณผลประเมิน", type="primary", key="run_evaluation"):
                    evaluated, evaluation = evaluate_labeled_dataframe(
                        eval_data, load_model(), bundle, metrics["selected_threshold"]
                    )
                    st.session_state["evaluation_result"] = (eval_hash, evaluated, evaluation)
                saved = st.session_state.get("evaluation_result")
                if saved and saved[0] == eval_hash:
                    _, evaluated, evaluation = saved
                    with st.container(horizontal=True):
                        st.metric("Accuracy", f"{evaluation['accuracy']:.3f}", border=True)
                        st.metric("Precision", f"{evaluation['precision']:.3f}", border=True)
                        st.metric("Recall", f"{evaluation['recall']:.3f}", border=True)
                        st.metric("F1-score", f"{evaluation['f1']:.3f}", border=True)
                        st.metric("ROC-AUC", f"{evaluation['roc_auc']:.3f}" if evaluation["roc_auc"] is not None else "คำนวณไม่ได้", border=True)
                    st.caption(f"ใช้ threshold เดิม {metrics['selected_threshold']:.2f} · {len(evaluated):,} รายการ")
                    matrix = evaluation["confusion_matrix"]
                    matrix_fig = px.imshow(
                        matrix,
                        text_auto=True,
                        x=["ทำนายไม่ล่าช้า", "ทำนายล่าช้า"],
                        y=["จริงไม่ล่าช้า", "จริงล่าช้า"],
                        color_continuous_scale=["#1d1d1d", "#ff6038"],
                        aspect="auto",
                        labels={"x": "ผลทำนาย", "y": "ผลจริง", "color": "จำนวน"},
                    )
                    st.markdown("##### Confusion Matrix")
                    st.plotly_chart(matrix_fig, width="stretch", alt="ตารางเปรียบเทียบผลจริงกับผลทำนาย")
                    if evaluation["roc_curve"] is not None:
                        fpr, tpr, _ = evaluation["roc_curve"]
                        roc_fig = go.Figure()
                        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC-AUC {evaluation['roc_auc']:.3f}", line={"color": "#ff6038"}))
                        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="สุ่ม", line={"color": "#777777", "dash": "dash"}))
                        roc_fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=360)
                        pr_precision, pr_recall, _ = evaluation["pr_curve"]
                        pr_fig = go.Figure(go.Scatter(x=pr_recall, y=pr_precision, mode="lines", name=f"AP {evaluation['average_precision']:.3f}", line={"color": "#ff6038"}))
                        pr_fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=360)
                        left, right = st.columns(2)
                        left.markdown("##### ROC Curve")
                        left.plotly_chart(roc_fig, width="stretch", alt="กราฟ ROC สำหรับข้อมูลประเมิน")
                        right.markdown("##### Precision–Recall Curve")
                        right.plotly_chart(pr_fig, width="stretch", alt="กราฟ Precision Recall สำหรับข้อมูลประเมิน")
                    else:
                        st.info("คำนวณ ROC/PR ไม่ได้ เพราะข้อมูลประเมินมีเพียงคลาสเดียว")
        except Exception as exc:
            st.error(f"ประเมิน CSV ไม่สำเร็จ: {exc}", title="ตรวจสอบข้อมูลประเมิน")

else:
    st.subheader("วิธีใช้")
    with st.container(border=True):
        st.markdown("#### ทำนายคำสั่งซื้อหนึ่งรายการ")
        st.write("เลือก **ทำนายความเสี่ยง → กรอกหนึ่งรายการ** แล้วกรอกข้อมูลให้ครบ กดปุ่มประเมินเพื่อดูผล")
    with st.container(border=True):
        st.markdown("#### ทำนายจากไฟล์ CSV")
        st.write("เลือก **ทำนายความเสี่ยง → อัปโหลด CSV หลายรายการ** ใช้ไฟล์ตัวอย่างดูชื่อคอลัมน์ แล้วเลือกไฟล์ของคุณ")
    with st.container(border=True):
        st.markdown("#### ดูภาพรวม")
        st.write("ไปที่ **ภาพรวม** แล้วอัปโหลด CSV ระบบจะประเมินรายการและแสดงกราฟความเสี่ยงตามวิธีจัดส่ง ตลาด แผนก และกลุ่มลูกค้า")
    with st.container(border=True):
        st.markdown("#### ตรวจว่าโมเดลแม่นแค่ไหน")
        st.write(f"ไปที่ **ผลโมเดล** แล้วอัปโหลดชุดข้อมูลที่ไม่เคยใช้ฝึก ซึ่งมีคอลัมน์ผลจริง `{TARGET_COLUMN}` เพื่อคำนวณ Accuracy, Recall, Confusion Matrix และกราฟ ROC/PR")
    st.info(
        f"เกณฑ์เตือนของโมเดลคือ {metrics['selected_threshold']:.0%}. ระดับต่ำ/กลาง/สูงเป็นการแบ่งคะแนนเพื่อให้อ่านง่าย ยังไม่ใช่สามคลาสที่ผ่านการฝึกและสอบเทียบแยกกัน; ช่วงกลางเริ่มที่เกณฑ์เตือน และช่วงสูงเริ่มที่ {display_high_threshold(metrics['selected_threshold']):.0%}.",
        title="ความหมายของระดับความเสี่ยง",
    )
    st.warning("ไฟล์ทดสอบสังเคราะห์ใช้ลองหน้าจอและขั้นตอนทำนายเท่านั้น ผลไม่ใช่สถิติออเดอร์จริงและใช้วัดความแม่นยำไม่ได้", title="หมายเหตุ")
