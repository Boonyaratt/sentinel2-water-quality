# ฟีเจอร์ใน Google Drive ใช้ตรงไหน

**อ่านรูปและโค้ดบน GitHub, ใช้ข้อมูลบน Drive เพื่อรันจริง** ฟีเจอร์ระดับสถานีเป็น CSV สำหรับสร้างตารางฝึก ส่วนฟีเจอร์ระดับแผนที่เป็น GeoTIFF สำหรับทำนายทีละพิกเซล ใช้แทนกันไม่ได้

| ชื่อโฟลเดอร์ต้นทางใน Drive | ใช้ทำอะไร | ตำแหน่งสำเนาใน ZIP 199 ไฟล์ |
|---|---|---|
| `rswq_exports_s2_chunks_epo14_2562_2568_b100_watermask_v4_s2formula` | Sentinel-2 CSV ที่ export แยกชิ้น นำมารวมและจับคู่กับตารางภาคสนาม | `data_ready/station_features/export_chunks/sentinel2_b100_v4/`; ตารางที่รวมแล้วอยู่ใน `data_ready/station_features/` |
| `rswq_context_features_ctx_2562_2568_gsmap_era5h_strict_v2` | CSV ข้อมูลแวดล้อมรายตัวอย่าง นำไปรวมกับ Sentinel-2 สำหรับฝึกโมเดล | `data_ready/station_features/export_chunks/context_strict_v2/` และ CSV ที่รวมแล้วใน `data_ready/station_features/` |
| `Water_Quality_RS_Moving_Window_Polygon` | GeoTIFF 62 แบนด์บนขอบเขตที่ตรวจไว้ เป็นอินพุตของ Colab 04 และ 05 (Fig. 3–4) | `data_ready/rasters/polygon/features/` |
| `Water_Quality_RS_Sentinel2_RGB_Polygon` | RGB สำหรับแผนที่ภาพดาวเทียมใน Colab 04; Colab 05 ใช้ภาพพื้นหลังจาก Esri | `data_ready/rasters/polygon/rgb/` |
| `Water_Quality_RS_Figure2_Features` และ `Water_Quality_RS_Figure2_RGB` | ฟีเจอร์ 62 แบนด์และภาพจริงของทะเลหลวง วันที่ 4 และ 19 มิ.ย. 2024 ใช้ใน Colab 06b (Fig. 5) | `data_ready/rasters/two_dates/features/` และ `rgb/` |

ชื่อคล้ายกันไม่ได้แปลว่าใช้แทนกันได้: โฟลเดอร์ `Water_Quality_RS_Moving_Window` ที่ไม่มี `_Polygon` และชุด `b200`/Phuket เป็นคนละชุดกับอินพุตหลักข้างบน ส่วน Regional SHAP (Fig. 7) ใช้กริด `regional_shap_v1` ที่แยกไว้ใน ZIP เต็ม อย่าเลือก GeoTIFF ด้วยชื่อไฟล์หรือวันที่อย่างเดียว

## ถ้าต้องการทำแผนที่จาก TIFF ที่มีใน Drive แล้ว

1. ขอ `Water_Quality_Map_Support.zip` จากเจ้าของงาน แล้วเพิ่ม shortcut ของไฟล์นี้ใน **My Drive** แพ็กเล็กนี้มี metadata, ขอบเขต, ตารางที่ Fig. 5 ต้องใช้ และเซลล์โหลดโมเดล ไม่มี TIFF และไม่ใช่ตัวแทนของชุดรับช่วง 199 ไฟล์
2. เปิด [Colab 04, 05 หรือ 06b](../notebooks/README.md) ตั้ง `USE_EXISTING_DRIVE_RASTERS = True` ในเซลล์แรก ถ้า ZIP อยู่ในโฟลเดอร์ย่อย ให้แก้ `ARCHIVE_PATH`
3. ถ้าโฟลเดอร์ TIFF ไม่อยู่ใต้ My Drive โดยตรง ให้เพิ่ม shortcut ของโฟลเดอร์ที่ได้รับสิทธิ์ หรือแก้ `DRIVE_RASTER_PATHS` ในเซลล์ถัดไป แล้วกด Run all โค้ดจะอ่านเฉพาะ TIFF ของรูปที่เลือกและตรวจ SHA-256 ก่อนรันโมเดล จึงไม่ต้อง export Earth Engine ใหม่

ถ้าจะส่งต่อให้คนเริ่มตั้งแต่ข้อมูลภาคสนาม ตรวจผลอ้างอิง หรือทำ SHAP ให้ใช้ **ZIP เต็ม 199 ไฟล์** เพราะการแชร์เฉพาะโฟลเดอร์ TIFF หรือแพ็กแผนที่เล็กไม่มีข้อมูลครบสำหรับขั้นตอนเหล่านั้น

## ถ้าต้องการดูโค้ดที่คำนวณฟีเจอร์

เปิด [สารบัญ Feature Generation](../notebooks/README.md#feature-generation-google-earth-engine) แล้วเลือกตามงาน:

| Notebook | ต้องเตรียม | ได้อะไร |
|---|---|---|
| 01 Sentinel-2 Station Features | CSV ภาคสนามที่คลีนแล้ว | CSV Sentinel-2 แบบรัศมี 100 เมตร, watermask v4 และตาราง strict/relaxed |
| 02 Environmental Context Features | CSV ภาคสนามที่คลีนแล้ว | CSV ข้อมูลแวดล้อมรุ่น `gsmap_era5h_strict_v2` |
| 03 Moving-Window Polygon Features | ตาราง coverage, context, polygon ที่ตรวจแล้ว; ตารางโมเดลสำหรับ audit เป็นตัวเลือก | feature GeoTIFF และ RGB บนพื้นที่ polygon |
| 03b Two-Date Feature Preparation | coverage, context, polygon และ GeoTIFF อ้างอิงกริด | คู่วัน, metadata และไฟล์ feature/RGB บนกริดเดียวกัน |

ไฟล์สาธารณะอ้างอิงต้นฉบับในชุดรับช่วง ตัดผลที่บันทึกไว้และเซลล์ QA ที่อ้างถึงตัวอย่างเฉพาะออก กรอก `EE_PROJECT` ของตัวเอง ตั้งพาธอินพุตเป็นรุ่นที่ตรวจแล้ว และเปิดสวิตช์ export เฉพาะเมื่อพร้อม หากทำ Fig. 5 เดิมให้ยึดคู่วันจาก manifest ที่ล็อกไว้; การรันจัดอันดับหรือค้นภาพใหม่ไม่ใช่เงื่อนไขสำหรับทำซ้ำรูปเดิม

ชื่อ `03web` / `04web` ใน Drive เป็นสายงานเว็บ ไม่ใช่ต้นทางในชุดทำซ้ำผลของบทความ และไม่ควรเลือกจากวันที่แก้ไขล่าสุดเพียงอย่างเดียว
