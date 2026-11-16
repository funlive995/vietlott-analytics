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
