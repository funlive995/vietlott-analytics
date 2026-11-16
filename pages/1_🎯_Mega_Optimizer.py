import json
import requests
import pandas as pd
import streamlit as st

from optimizer import (
    walk_forward_backtest, generate_portfolio,
    theoretical_jackpot_probability, portfolio_coverage
)

DATA_URL = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl"
st.set_page_config(page_title="Mega 6/45 Optimizer", page_icon="🎯", layout="wide")

@st.cache_data(ttl=3600)
def load_draws():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    rows = []
    for line in response.text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        nums = item.get("result", [])
        if len(nums) == 6 and len(set(nums)) == 6 and all(1 <= int(n) <= 45 for n in nums):
            rows.append({"date": item["date"], "draw_id": str(item.get("id", "")),
                         "numbers": sorted(int(n) for n in nums)})
    df = pd.DataFrame(rows).drop_duplicates(["draw_id", "date"])
    return df.sort_values(["date", "draw_id"]).reset_index(drop=True)

st.title("🎯 Mega 6/45 — Optimizer & Out-of-sample Lab")
st.caption("Walk-forward backtest · shrinkage · portfolio construction · jackpot probability")

try:
    df = load_draws()
except Exception as exc:
    st.error(f"Không tải được dữ liệu lịch sử: {exc}")
    st.stop()

st.info(
    f"Đã tải **{len(df):,} kỳ** ({df.date.iloc[0]} → {df.date.iloc[-1]}). "
    "Backtest theo thứ tự thời gian: mỗi kỳ chỉ được dự báo bằng dữ liệu trước kỳ đó."
)

tab1, tab2, tab3 = st.tabs(["🧪 Kiểm định ngoài mẫu", "🎟️ Tạo danh mục 50.000đ", "📐 Xác suất Jackpot"])

with tab1:
    st.subheader("Walk-forward backtest")
    c1, c2 = st.columns(2)
    warmup = c1.slider("Số kỳ warm-up", 50, min(500, max(50, len(df)-2)), min(100, len(df)-2), 10)
    recent = c2.slider("Cửa sổ gần nhất", 20, 300, 100, 10)
    if st.button("Chạy backtest", type="primary"):
        with st.spinner("Đang chạy kiểm định từng kỳ..."):
            scores, summary = walk_forward_backtest(df["numbers"].tolist(), warmup, recent)
        st.session_state["wf_scores"] = scores
        st.session_state["wf_summary"] = summary

    if "wf_summary" in st.session_state:
        summary = st.session_state["wf_summary"].copy()
        display = summary.copy()
        for col in ["mean_log_score", "uniform_mean_log_score", "delta_vs_uniform", "delta_se", "z_vs_uniform"]:
            display[col] = display[col].map(lambda x: f"{x:.5f}")
        display["win_rate"] = (display["win_rate"] * 100).map(lambda x: f"{x:.1f}%")
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.caption("Delta log-score > 0 là tốt hơn uniform; z chỉ là chỉ báo paired, không thay thế kiểm định nhiều giả thuyết.")
        best = summary.sort_values("delta_vs_uniform", ascending=False).iloc[0]
        if best["delta_vs_uniform"] > 0 and best["z_vs_uniform"] > 2:
            st.success("Có tín hiệu dự báo dương trong mẫu kiểm định này. Cần xác nhận trên giai đoạn tương lai mới.")
        else:
            st.warning("Chưa có bằng chứng đủ mạnh rằng mô hình tần suất dự báo tốt hơn uniform. Không tăng xác suất Jackpot dựa trên các điểm số này.")
        st.line_chart(st.session_state["wf_scores"][["uniform", "shrunk_frequency", "recency_blend"]])
        st.download_button("Tải kết quả từng kỳ CSV",
                           st.session_state["wf_scores"].to_csv(index=False).encode("utf-8-sig"),
                           "mega645_walk_forward.csv", "text/csv")
    else:
        st.write("Nhấn **Chạy backtest** để kiểm định mô hình tần suất co rút và mô hình kết hợp xu hướng gần đây so với baseline uniform.")

with tab2:
    st.subheader("Danh mục vé theo ngân sách")
    budget = st.number_input("Ngân sách mỗi kỳ (đồng)", min_value=10000, max_value=1000000,
                             value=50000, step=10000)
    strategy = st.radio("Chiến lược", ["Đồng đều / không giả định bias", "Thử nghiệm theo tần suất (không chứng minh lợi thế)"],
                        horizontal=True)
    if st.button("Tạo danh mục vé", type="primary"):
        mode = "frequency_experimental" if strategy.startswith("Thử nghiệm") else "diversified"
        portfolio = generate_portfolio(df["numbers"].tolist(), int(budget), strategy=mode)
        st.session_state["portfolio"] = portfolio
    if "portfolio" in st.session_state:
        portfolio = st.session_state["portfolio"]
        st.dataframe(portfolio[["Vé", "Số"]], hide_index=True, use_container_width=True)
        coverage = portfolio_coverage(portfolio)
        lines = len(portfolio)
        p = theoretical_jackpot_probability(lines)
        a, b, c = st.columns(3)
        a.metric("Số vé khác nhau", f"{lines}")
        b.metric("Số cặp được phủ", f"{coverage['pairs_covered']:,}")
        c.metric("Xác suất Jackpot", f"1 / {1/p:,.0f}")
        st.download_button("Tải danh mục CSV",
                           portfolio[["Vé", "Số"]].to_csv(index=False).encode("utf-8-sig"),
                           "mega645_portfolio.csv", "text/csv")
        st.caption("Hai chiến lược có cùng xác suất Jackpot lý thuyết nếu kết quả quay công bằng và các vé đều khác nhau.")
    st.warning("Không có chiến lược nào được đảm bảo thắng. Chỉ chi tiêu số tiền có thể mất.")

with tab3:
    st.subheader("Xác suất Jackpot — toán học, không phải dự đoán")
    lines = st.slider("Số vé khác nhau", 1, 100, 5)
    p = theoretical_jackpot_probability(lines)
    st.metric("Xác suất trúng ít nhất một Jackpot", f"{p*100:.8f}%")
    st.write(f"Tương đương khoảng **1 trên {1/p:,.0f}** cho {lines} vé khác nhau.")
    st.write(f"Tổng số tổ hợp 6/45: **8.145.060**.")
    st.write("Với 5 vé khác nhau, xác suất là 5/8.145.060 = 1/1.629.012.")
    st.caption("Tần suất lịch sử, số nóng/lạnh, tổng, chẵn/lẻ hay cặp số không làm thay đổi xác suất của một tổ hợp cụ thể nếu mỗi kỳ quay độc lập và công bằng.")

st.divider()
st.caption("Research tool only. Backtest là kiểm định lịch sử ngoài mẫu theo thời gian, không phải cam kết lợi nhuận hay dự đoán kết quả.")
