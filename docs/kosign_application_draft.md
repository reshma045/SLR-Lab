# KoSign data application: draft for the supervisor

KoSign (工学院大学 多用途型日本手話言語データベース) is distributed by the NII Informatics Research Data
Repository (IDR): https://www.nii.ac.jp/dsc/idr/rdata/KoSign/

**What it contains:** about 6,300 JSL words and 10 dialogue themes, recorded by native Deaf signers.
Data: front/side 4K or Full HD video (MXF/MP4), optical motion capture (BVH/C3D/FBX), and Kinect v2
depth. Phase 1 has 3,701 words; phase 2 has 1,172 words with variants (2,655 motions). Size is about
2.5 TB for words and 0.4 TB for dialogues (compressed). For this project, the **front-camera MP4**
files are enough.

**Conditions (from the IDR page; check the current usage policy before applying):**
- Research use only. Mainly for researchers at universities and public research institutions.
- The research representative must be **full-time faculty**. Students cannot apply alone.
- The consent form needs signature and seal from someone with signing authority (typically
  **dean level or above**).
- Only members of the same organisation working directly on the research may use the data.
- Notify IDR before media publicity, and submit an annual usage report for publications.
- Free of charge.

**Steps:**
1. Read Kogakuin University's KoSign usage policy and the IDR service policy (linked from the page),
   then fill in the Word application form downloaded from the page.
2. Email the form to the IDR secretariat (address on the IDR page) with the subject:
   `工学院大学多用途型日本手話言語データベース利用申請（会津大学）`
3. After review, IDR sends the policies and consent form. Sign and seal it, then post it to IDR.
4. Data is provided for download after confirmation. Plan for ~3 TB of storage, or ask whether a
   front-video-only subset can be provided.

---

## Draft email to IDR (for the supervisor to adapt and send)

件名：工学院大学多用途型日本手話言語データベース利用申請（会津大学）

国立情報学研究所 情報学研究データリポジトリ（IDR）事務局 御中

会津大学 〇〇〇〇研究室の〇〇〇〇と申します。

当研究室では、Webカメラ映像から日本手話の単語をリアルタイムに認識するシステム（MediaPipe による
手・上半身・顔（口形・視線・眉）の特徴点抽出と、深層学習による孤立単語認識）の研究を行っております。
現在は単一話者の参照映像50語で学習しておりますが、話者に依存しない認識精度の評価・向上のため、
貴リポジトリで提供されている「工学院大学多用途型日本手話言語データベース（KoSign）」を研究目的で
利用させていただきたく、利用申請書を添付にてお送りいたします。

利用目的：日本手話の単語認識モデルの学習・評価（修士論文研究を含む学術研究）
利用データ：単語データ（正面カメラ映像を中心に）、必要に応じて対話データ
利用者：〇〇〇〇（研究代表者・教員）、〇〇〇〇（大学院生）

データの利用にあたっては、工学院大学の利用方針および IDR のサービスポリシーを遵守いたします。
ご審査のほど、どうぞよろしくお願い申し上げます。

会津大学 〇〇学部 〇〇〇〇
メール：〇〇〇〇@u-aizu.ac.jp

---

## When the data arrives

Write a CSV that maps each video file to its sign word (columns `file,sign,english`), then run:

```powershell
python scripts/import_videos.py --source kosign --dir D:\KoSign\front --map kosign_map.csv            # only our 50 words
python scripts/import_videos.py --source kosign --dir D:\KoSign\front --map kosign_map.csv --add-new  # + new words
python run_training.py
```

Imported clips become extra training examples. With `--add-new` they also extend the vocabulary.
The dashboard picks the new words up automatically.
