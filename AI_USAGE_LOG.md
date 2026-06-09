# AI Usage Log — moby/moby Statistical Audit

## Summary

| Member | Peran | Tools | ~% Kode AI-assisted | Interpretation AI-assisted? |
|--------|-------|-------|--------------------|-----------------------------|
| Muhammad Raffy Dwiputra | Data Engineer | Claude, Gemini | 55 % | No |
| Muhammad Yusuf Jiddan | Estimation Analyst | Claud | 65 % | No |
| Dhida Framudya Wiradonna | Inference Analyst | Claude, Gemini | 50 % | No |
| Malik Alfat Muzaki | Hypothesis Analyst | Claude | 50 % | No |
| Audylia Aska Widiaputri | Computation Analyst | Claude | 55 % | No |

## Per-Member Detail

### Member A — Muhammad Raffy Dwiputra
| # | Task | Tool | Prompt (ringkas) | Cara output digunakan |
|---|------|------|------------------|----------------------|
| 1 | Data Collecting | Claude | Tolong buatkan code cleaning data untuk file raw yang saya berikan | Mengambil referensi code untuk cleaning data |

### Member B — Muhammad Yusuf Jiddan
| # | Task | Tool | Prompt (ringkas) | Cara output digunakan |
|---|------|------|------------------|----------------------|
| 1 | Estimation py | Claude | Bantu buatkan dan jabarkan estimator py sesuai file panduan yang saya kasih | Untuk memahami role yang diambil dan dijadikan sebagai referensi |

### Member C — Dhida Framudya Wiradonna Handled By. Member A
| # | Task | Tool | Prompt (ringkas) | Cara output digunakan |
|---|------|------|------------------|----------------------|
| 1 | Referensi Peggunaan Scipy | Claude | Cara ambil nilai Z kritis dan Kuantil dari Scipy | Mengetahui nama fungsi yang tepat dan sintaks fungsi pada logika α=k+1 dan β=m+1 diverifikasi mandiri dari Tsun p.269 |

### Member D — Malik Alfat Muzaki
| # | Task | Tool | Prompt (ringkas) | Cara output digunakan |
|---|------|------|------------------|----------------------|
| 1 | Code Generate | Claude | Buat format code Python untuk one-sample Z-test dan two-sample Z-test | Dasar Kode sebelum saya kembangkan sesuai data yang saya miliki | 

### Member E — Audylia Aska Widiaputri
| # | Task | Tool | Prompt (ringkas) | Cara output digunakan |
|---|------|------|------------------|----------------------|
| 1 | Diskusi pendekatan simulasi | Claude | "Metode Monte Carlo yang cocok untuk dataset issue tracking?" | Dijadikan referensi pemilihan metode, disesuaikan dengan dataset |

## Group Reflection (150–300 kata)
How did your group's use of AI evolve over three weeks? What did AI handle well?
Where did output need significant correction? Was there a moment you chose _not_
to use AI — and why?

Jawab:
Perkembangan penggunaan AI kami menjadi lebih selektif. Jika pada awalnya AI digunakan untuk memperoleh contoh dan referensi umum, pada tahap akhir AI lebih banyak digunakan untuk memeriksa logika, memvalidasi implementasi. AI membantu menyelesaikan tugas tugas teknis dan repetitive, seperti memberikan contoh sintaks Python, menjelaskan penggunaan library statistik, menghasilkan kerangka kode awal, serta membantu memahami konsep statistik yang digunakan dalam proyek. Namun ada bagian yang kami memutuskan tidak menggunakkan AI seperti untuk logika berfikir dasar, seperti contoh menentukan RQ, karena jika menggunakkan AI, AI akan menggenerate secara halusinasi dan tidak sesuai konteks dari data yang kita ambil. Selain itu ada bagian bagian yang harus kita crosscheck dan berikan konteks ke AI terlebih dahulu, seperti jika kita asal menggenerate kerangka code maka AI akan membuat code tanpa mengetahui spesifikasi yang diminta tugas ini, jadi kita perlu crosscheck ulang.
