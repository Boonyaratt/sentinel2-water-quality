# คู่มือรับช่วงงาน

เส้นทางหลักเริ่มที่ **ตารางภาคสนามที่คลีนแล้ว** และใช้ไฟล์ดาวเทียมที่คำนวณไว้ใน Drive ของทีม รันบนเครื่องตัวเองได้โดยไม่ต้องมีสิทธิ์ Earth Engine

## 1. เตรียมโฟลเดอร์

1. ขอไฟล์ส่งต่อจากเจ้าของโครงการ ใช้ `Water_Quality_Processed_Handoff.zip` แบบไฟล์เดียว หรือชุด Drive สองส่วน `Water_Quality_Handoff_01_Core.zip` และ `Water_Quality_Handoff_02_Two_Date_Rasters.zip`
2. แตกไฟล์ลงโฟลเดอร์แยก หากใช้ชุดสองส่วน ให้แตก **ทั้งสองไฟล์ลงโฟลเดอร์เดียวกัน** เพื่อให้ `data_ready/rasters/two_dates/` อยู่ร่วมกับข้อมูลชุดหลัก เปิด terminal ในโฟลเดอร์ที่เห็น `START_HERE.md`, `data_ready/`, `models/`, `src/` และ `tools/`
3. สร้าง virtual environment ด้วย Python 3.11 และติดตั้ง requirements ของชุด

```powershell
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -B tools/reproduce.py --output run_output/first_check
```

บน macOS หรือ Linux ให้ใช้ `./.venv/bin/python` แทน `./.venv/Scripts/python.exe`

ดู `run_output/first_check/verification.json` ควรผ่าน `assembly`, `locked_models`, `spatial_inputs_and_samples` และ `spatial_comparisons` ส่งไฟล์นี้กลับให้ผู้ส่งมอบเมื่อเริ่มรันจากเครื่องตัวเองได้แล้ว

## 2. เลือกจุดเริ่ม

| ต้องการทำอะไร | เปิดไฟล์หรือโฟลเดอร์ไหน | ผลที่ควรได้ |
|---|---|---|
| อ่านและตรวจข้อมูลภาคสนาม | `data_ready/ground/wq_14_2562_2568_rs_ready.csv` เปิดด้วย Excel ได้ | ตารางคลีน 826 × 116 และตารางตรวจพิกัดในโฟลเดอร์เดียวกัน |
| รวม Sentinel-2 กับ context | `data_ready/station_features/` + `src/03_assembly/02_Merge_rswq_dataset.ipynb` หรือใช้ตัวรัน | relaxed table 340 × 226 ตรงกับ `data_ready/model/` |
| ตรวจ Table 5 | `src/04_models/` และ `expected_outputs/Table05/` | ใช้โน้ตบุ๊ก Turbidity `before_artifact_export` เซลล์ 24–32 สำหรับตาราง และใช้โน้ตบุ๊กที่ไม่มี suffix สำหรับ LinearSVR mapping artifacts |
| ทำแผนที่และเปรียบเทียบผล | `src/05_mapping/`, `src/06_two_dates/`, `data_ready/rasters/` และ `COLAB.md` | Fig. 3–4 ใช้ polygon; Fig. 5 ใช้ two_dates และ common mask |
| ทำ SHAP | `src/07_xai/`, `data_ready/model/`, `models/` และ raster `regional_shap_v1` | เพิ่ม `--global-shap` เพื่อสร้างตารางและรูป Global SHAP; `--regional-shap` สำหรับการคำนวณทุกพิกเซลซึ่งใช้เวลานานกว่า |

## 3. ก่อนเริ่มแก้ผล

ใช้ `PIPELINE.md` เพื่อตามจากรูปหรือตารางกลับไปยังข้อมูลเข้า และ `MANIFEST.csv` เพื่อตรวจรุ่นกับ SHA-256. เขียนผลใหม่ใน `run_output/` และเก็บ reference outputs ไว้สำหรับเปรียบเทียบ

ชุดนี้ให้ใช้ไฟล์ GEE ที่ export ไว้แล้ว การรันใหม่ตั้งแต่ดาวน์โหลดภาพหรือสร้าง raster ใหม่เป็นอีกเส้นทางหนึ่งซึ่งต้องใช้ Earth Engine. ส่วน Excel ก่อนคลีนอยู่นอกขอบเขตของชุดสำหรับผู้รับช่วงนี้
