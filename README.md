# Vietlott Analytics — Mega 6/45

Dashboard Streamlit phân tích lịch sử Mega 6/45.

## Chạy thử tại máy
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Xuất bản website
1. Đăng nhập [Streamlit Community Cloud](https://share.streamlit.io/) bằng GitHub.
2. Chọn **Create app**.
3. Repository: `funlive995/vietlott-analytics`, branch: `main`, file: `app.py`.
4. Nhấn **Deploy**. Sau khi Streamlit hoàn tất, anh sẽ có URL website riêng.

## Dữ liệu
Ứng dụng đọc dữ liệu Mega 6/45 dạng JSONL từ:
https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl

Nguồn: https://github.com/vietvudanh/vietlott-data (MIT). Dữ liệu cache 1 giờ và có nút làm mới.

## Chức năng
- Tổng quan số kỳ và bộ lọc theo kỳ/ngày
- Tần suất 45 số, top số nhiều/ít
- Tần suất cặp số và heatmap
- Phân phối chẵn/lẻ, tổng 6 số, số liên tiếp
- Lịch sử kỳ quay và xuất CSV

Lưu ý: thống kê quá khứ không dự đoán hay bảo đảm kết quả tương lai.


## Optimizer & kiểm định ngoài mẫu
Dashboard có thêm trang **Mega Optimizer** trong thanh điều hướng:
- Walk-forward backtest với warm-up mặc định 100 kỳ, kiểm tra 1.292 kỳ tiếp theo trên bộ dữ liệu 1.392 kỳ.
- Baseline uniform so sánh với frequency shrinkage và mô hình kết hợp tần suất gần đây.
- Log-score xác suất của **tập hợp 6 số không có thứ tự**, tính chính xác bằng dynamic programming để cộng đủ 6! thứ tự rút theo Plackett–Luce; paired delta và tải CSV kết quả.
- Tạo danh mục vé theo ngân sách; mặc định không giả định có bias.
- Tính xác suất Jackpot lý thuyết từ số vé khác nhau.

Engine nằm trong `optimizer.py`. Các mô hình thống kê chỉ là thử nghiệm; nếu không vượt baseline ngoài mẫu thì không được diễn giải thành lợi thế dự báo.


## Nghiên cứu nâng cao: 3 hướng
Trang **Bias & Monte Carlo** bổ sung:
1. **Combinatorial portfolio:** tạo 5 vé không trùng số, đạt 30 số riêng, 75 cặp riêng và 100 bộ ba riêng. Đây là tối đa về độ phủ số/cặp/bộ ba trong cấu hình 5 vé.
2. **Bias diagnostics:** exact binomial tests cho 45 số và 990 cặp, điều chỉnh đa kiểm định bằng Benjamini–Hochberg (BH) và Benjamini–Yekutieli (BY); so sánh ổn định giữa hai nửa lịch sử và kiểm tra giao nhau giữa hai kỳ liên tiếp.
3. **Monte Carlo:** so sánh danh mục random và disjoint dưới giả thuyết quay công bằng; báo độ phủ và xác suất có ít nhất một vé đạt >=2, >=3, >=4, >=5 hoặc 6 số.

Monte Carlo không đủ lớn để đo xác suất Jackpot (cỡ 1/1,629,012 cho 5 vé). Xác suất Jackpot phải dùng công thức chính xác, không ước lượng từ số lần mô phỏng hiếm.
