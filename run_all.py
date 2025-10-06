import subprocess, json, os
os.makedirs("results", exist_ok=True)

jobs = [
    ("dataset\\CICDDoS2017", True,  "results\\cic2017_fs.json"),
    ("dataset\\CICDDoS2019", False, "results\\cic2019_multiclass_fs.json"),
    ("dataset\\UNSW_NB15",   True,  "results\\unsw_fs.json"),
    ("dataset\\NSL-KDD\\KDDTrain+.arff", True, "results\\nsl_fs.json"),
]

base = ["python","ids_benchmark_v2.py","--test-size","0.2",
        "--drop-constant","--drop-corr","0.98","--mi-topk","40"]

summary=[]
for path,binary,out in jobs:
    cmd = base + ["--input", path, "--save", out]
    if binary: cmd.append("--binary")
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    with open(out,"r",encoding="utf-8") as f: summary.append(json.load(f))

with open("results\\summary_all.json","w",encoding="utf-8") as f:
    json.dump(summary,f,ensure_ascii=False,indent=2)
print("[OK] Đã lưu: results\\summary_all.json")
