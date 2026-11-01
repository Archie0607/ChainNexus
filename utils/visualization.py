import plotly.express as px
import pandas as pd

COLORS = ["#29d3e2", "#3478f6", "#ff6b7a", "#9b8afb", "#54d6a1"]

def risk_pie(df):
    counts = df["Risk Level"].value_counts().rename_axis("Risk Level").reset_index(name="Orders")
    chart = px.pie(counts, names="Risk Level", values="Orders", hole=.72, color="Risk Level",
                   color_discrete_map={"สูง":"#ff6038", "กลาง":"#ffad45", "ต่ำ":"#777777"})
    chart.update_traces(textposition="outside", textinfo="percent")
    chart.update_layout(
        height=300,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"{len(df):,}<br><span style='font-size:12px'>รายการ</span>", x=.5, y=.5, showarrow=False, font=dict(size=22))],
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="center", x=.5),
    )
    return chart

def grouped_risk(df, column):
    data = df.groupby(column, dropna=False).agg(Orders=("Risk Classification", "size"), Risk_Rate=("Risk Classification", "mean")).reset_index()
    data["Risk Rate (%)"] = data["Risk_Rate"] * 100
    data = data.sort_values("Risk Rate (%)", ascending=True)
    chart = px.bar(data, x="Risk Rate (%)", y=column, orientation="h",
                   hover_data=["Orders"], color_discrete_sequence=["#ff6038"],
                   category_orders={column: data[column].astype(str).tolist()})
    chart.update_xaxes(range=[0, 100], ticksuffix="%", title="สัดส่วนเสี่ยงสูง", gridcolor="#343434")
    chart.update_yaxes(title=None, automargin=True)
    chart.update_layout(height=300, showlegend=False, margin=dict(l=8, r=12, t=8, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return chart
