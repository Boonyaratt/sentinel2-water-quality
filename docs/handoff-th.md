# คู่มือรับช่วงงาน

เส้นทางหลักเริ่มที่ **ตารางภาคสนามที่คลีนแล้ว** และใช้ไฟล์ดาวเทียมที่คำนวณไว้แล้ว รันได้บน Colab หรือบนเครื่องตัวเองโดยไม่ต้องมีสิทธิ์ Earth Engine ดู [สารบัญ Colab](../notebooks/README.md) และ [รายการข้อมูลกับโน้ตบุ๊กต้นทาง](data-inventory.md) ประกอบ

ถ้าใช้ AI agent ให้เริ่มจาก [AGENTS.md](../AGENTS.md) ซึ่งมีสำเนาอยู่ในรากของ ZIP เต็ม v2

## เริ่มบน Colab

1. ขอ ZIP จากเจ้าของโครงการ แล้วเพิ่ม shortcut ไว้ใน My Drive หรืออัปโหลดสำเนาของตัวเอง
2. เปิด [00 Start Here](https://colab.research.google.com/github/Boonyaratt/sentinel2-water-quality/blob/main/notebooks/00_Start_Here.ipynb) กด Run all และอนุญาตให้ Colab อ่าน Drive ถ้าวาง ZIP ไว้ในโฟลเดอร์ย่อย ให้แก้ `ARCHIVE_PATH` ในเซลล์แรก
3. ส่ง `verification.json` กลับให้ผู้ส่งมอบ แล้วเลือก Colab 04, 05 หรือ 06b จากสารบัญตามรูปที่ต้องการ แต่ละโน้ตบุ๊กแตก ZIP และตั้งพาธให้เอง

## 1. เตรียมโฟลเดอร์

1. ขอ `Water_Quality_Processed_Handoff_Windows_v2.zip` จากเจ้าของโครงการ ชุดนี้เปิดใน File Explorer ได้และมีคู่มือสำหรับ AI agent ส่วน ZIP เต็มรุ่นก่อนยังใช้กับ Colab ได้
2. แตก ZIP ลงโฟลเดอร์แยก แล้วเปิด terminal ในโฟลเดอร์ที่เห็น `START_HERE.md`, `data_ready/`, `models/`, `src/` และ `tools/`
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
