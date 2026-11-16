import json
from itertools import combinations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

DATA_URL = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl"

st.set_page_config(page_title="Vietlott Analytics | Mega 6/45", page_icon="🎱", layout="wide")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    rows = []
    for line in response.text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        nums = item.get("result")
        if isinstance(nums, list) and len(nums) == 6:
            nums = sorted(int(n) for n in nums)
            if len(set(nums)) == 6 and all(1 <= n <= 45 for n in nums):
                rows.append({"date": pd.to_datetime(item["date"]).date(),
                             "draw_id": str(item.get("id", "")), "numbers": nums})
    df = pd.DataFrame(rows).drop_duplicates(subset=["draw_id", "date"])
    if df.empty:
        raise ValueError("Nguồn dữ liệu không trả về bản ghi hợp lệ.")
    return df.sort_values(["date", "draw_id"]).reset_index(drop=True)

def frequency_table(draws):
    counts = {n: 0 for n in range(1, 46)}
    for nums in draws["numbers"]:
        for n in nums:
            counts[n] += 1
    total = len(draws)
    return pd.DataFrame({"Số": list(counts), "Số lần xuất hiện": list(counts.values()),
                         "Tỷ lệ kỳ quay (%)": [counts[n] / total * 100 for n in counts]})

def pair_table(draws):
    counts = {}
    for nums in draws["numbers"]:
        for pair in combinations(nums, 2):
            counts[pair] = counts.get(pair, 0) + 1
    return pd.DataFrame([{"Số 1": a, "Số 2": b, "Số lần cùng xuất hiện": c}
                         for (a, b), c in counts.items()]).sort_values(
        ["Số lần cùng xuất hiện", "Số 1", "Số 2"], ascending=[False, True, True]
    ).reset_index(drop=True)

st.title("🎱 Vietlott Analytics")
st.caption("Mega 6/45 · Phân tích thống kê lịch sử · Dữ liệu cập nhật từ nguồn công khai")

with st.sidebar:
    st.header("Điều khiển")
    if st.button("↻ Tải lại dữ liệu mới nhất", use_container_width=True):
        load_data.clear()
        st.rerun()
    st.caption("Dữ liệu cache tối đa 1 giờ.")
    st.markdown("**Nguồn dữ liệu**")
    st.markdown("[vietvudanh/vietlott-data](https://github.com/vietvudanh/vietlott-data)")
    st.caption("Dữ liệu NDJSON công khai, giấy phép MIT.")

try:
    with st.spinner("Đang tải lịch sử Mega 6/45..."):
        all_draws = load_data()
except Exception as exc:
    st.error("Không tải được dữ liệu. Anh thử tải lại sau nhé.")
    st.code(str(exc))
    st.stop()

min_date, max_date = all_draws.date.min(), all_draws.date.max()
st.caption(f"Dữ liệu đến {max_date:%d/%m/%Y} · {len(all_draws):,} kỳ hợp lệ · Kỳ đầu {min_date:%d/%m/%Y}")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tổng số kỳ", f"{len(all_draws):,}")
k2.metric("Ngày bắt đầu", min_date.strftime("%d/%m/%Y"))
k3.metric("Ngày mới nhất", max_date.strftime("%d/%m/%Y"))
k4.metric("Số có thể xuất hiện", "1–45")

st.subheader("Bộ lọc thời gian")
preset = st.radio("Khoảng dữ liệu", ["Toàn bộ lịch sử", "30 kỳ gần nhất", "50 kỳ gần nhất",
    "100 kỳ gần nhất", "200 kỳ gần nhất", "500 kỳ gần nhất", "Tự chọn ngày"], horizontal=True)
if preset == "Tự chọn ngày":
    date_range = st.date_input("Chọn khoảng ngày", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d1, d2 = date_range
        draws = all_draws[(all_draws.date >= d1) & (all_draws.date <= d2)].copy()
    else:
        draws = all_draws.iloc[0:0].copy()
elif preset == "Toàn bộ lịch sử":
    draws = all_draws.copy()
else:
    draws = all_draws.tail(int(preset.split()[0])).copy()

if draws.empty:
    st.warning("Không có kỳ quay trong khoảng ngày đã chọn.")
    st.stop()

freq, pairs = frequency_table(draws), pair_table(draws)
st.caption(f"Đang phân tích {len(draws):,} kỳ, từ {draws.date.min():%d/%m/%Y} đến {draws.date.max():%d/%m/%Y}.")
tab1, tab2, tab3, tab4 = st.tabs(["📊 Tần suất số", "🔗 Cặp số", "📈 Phân phối", "🗂️ Lịch sử"])

with tab1:
    left, right = st.columns([1.5, 1])
    with left:
        fig = px.bar(freq.sort_values("Số"), x="Số", y="Số lần xuất hiện",
                     hover_data={"Tỷ lệ kỳ quay (%)": ":.2f"}, title="Tần suất xuất hiện của 45 số")
        fig.update_layout(xaxis=dict(dtick=1), height=470)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown("**Top 10 xuất hiện nhiều nhất**")
        st.dataframe(freq.sort_values(["Số lần xuất hiện", "Số"], ascending=[False, True]).head(10),
                     hide_index=True, use_container_width=True)
        st.markdown("**Top 10 xuất hiện ít nhất**")
        st.dataframe(freq.sort_values(["Số lần xuất hiện", "Số"]).head(10),
                     hide_index=True, use_container_width=True)
    st.download_button("Tải bảng tần suất CSV", freq.to_csv(index=False).encode("utf-8-sig"),
                       "mega645_frequency.csv", "text/csv")

with tab2:
    st.markdown("Thống kê số lần hai số cùng xuất hiện trong một kỳ (990 cặp có thể có).")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Top 20 cặp phổ biến**")
        st.dataframe(pairs.head(20), hide_index=True, use_container_width=True)
    with c2:
        pivot = pd.DataFrame(0, index=range(1, 46), columns=range(1, 46))
        for row in pairs.itertuples(index=False):
            pivot.loc[row[0], row[1]] = row[2]
            pivot.loc[row[1], row[0]] = row[2]
        heat = go.Figure(go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index),
                                    colorscale="Blues", hoverongaps=False))
        heat.update_layout(title="Heatmap đồng xuất hiện", height=700,
                           xaxis_title="Số", yaxis_title="Số")
        st.plotly_chart(heat, use_container_width=True)
    st.download_button("Tải thống kê cặp số CSV", pairs.to_csv(index=False).encode("utf-8-sig"),
                       "mega645_pairs.csv", "text/csv")

with tab3:
    stats = []
    for row in draws.itertuples(index=False):
        nums = row.numbers
        odd = sum(n % 2 for n in nums)
        stats.append({"Số lẻ": odd, "Số chẵn": 6 - odd, "Tổng 6 số": sum(nums),
                      "Số nhỏ (1–22)": sum(n <= 22 for n in nums),
                      "Số lớn (23–45)": sum(n >= 23 for n in nums),
                      "Có số liên tiếp": any(b - a == 1 for a, b in zip(nums, nums[1:]))})
    stat_df = pd.DataFrame(stats)
    a, b = st.columns(2)
    with a:
        odd_counts = stat_df["Số lẻ"].value_counts().reindex(range(7), fill_value=0).sort_index()
        fig = px.bar(x=[f"{i} lẻ / {6-i} chẵn" for i in range(7)], y=odd_counts.values,
                     labels={"x": "Cơ cấu chẵn/lẻ", "y": "Số kỳ"}, title="Phân phối chẵn/lẻ")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.plotly_chart(px.histogram(stat_df, x="Tổng 6 số", nbins=25, title="Phân phối tổng 6 số"),
                        use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Tổng trung bình", f"{stat_df['Tổng 6 số'].mean():.1f}")
    b.metric("Kỳ có số liên tiếp", f"{stat_df['Có số liên tiếp'].sum():,}")
    c.metric("Tỷ lệ có số liên tiếp", f"{stat_df['Có số liên tiếp'].mean()*100:.1f}%")

with tab4:
    hist = draws.copy()
    hist["Bộ số"] = hist.numbers.apply(lambda nums: " – ".join(f"{n:02d}" for n in nums))
    hist = hist[["draw_id", "date", "Bộ số"]].rename(columns={"draw_id": "Mã kỳ", "date": "Ngày quay"})
    st.dataframe(hist.sort_values("Ngày quay", ascending=False), hide_index=True, use_container_width=True)
    st.download_button("Tải lịch sử CSV", hist.to_csv(index=False).encode("utf-8-sig"),
                       "mega645_history.csv", "text/csv")

st.divider()
st.caption("Lưu ý: Các kỳ quay độc lập. Thống kê quá khứ chỉ mô tả dữ liệu, không dự đoán hay bảo đảm kết quả tương lai.")
