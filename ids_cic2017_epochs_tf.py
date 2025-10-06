# # import argparse, time, os, glob, json, io
# # import numpy as np
# # import pandas as pd
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# # from sklearn.feature_selection import mutual_info_classif
# # import lightgbm as lgb

# # LABEL_CANDS = ["Label","label","Attack","attack","attack_cat","class","Class","target","category","Category","result"]
# # TIME_CANDS  = ["Timestamp","timestamp","Flow ID","FlowID","StartTime","Start Time","stime","time","Date","datetime"]

# # def safe_read_csv(path: str) -> pd.DataFrame:
# #     # thử header + sep tự động và nhiều encoding
# #     encodings = ["utf-8-sig","utf-8","cp1252","latin1"]
# #     for enc in encodings:
# #         try:
# #             return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
# #         except Exception:
# #             continue

# #     with open(path, "rb") as f:
# #         data = f.read().decode("latin1", errors="ignore")
# #     return pd.read_csv(io.StringIO(data), sep=None, engine="python", on_bad_lines="skip")

# # def read_file(f: str) -> pd.DataFrame:
# #     ext = os.path.splitext(f)[1].lower()
# #     if ext == ".parquet":
# #         return pd.read_parquet(f)
# #     if ext == ".csv":
# #         return safe_read_csv(f)
# #     if ext == ".arff":
# #         import liac_arff # type: ignore
# #         with open(f, 'r', errors="ignore") as fp:
# #             arff = liac_arff.load(fp)
# #         return pd.DataFrame(arff['data'], columns=[c[0] for c in arff['attributes']])
# #     raise ValueError(f"Định dạng chưa hỗ trợ: {ext}")

# # def read_any(path: str) -> pd.DataFrame:
# #     if os.path.isdir(path):
# #         files = sorted(glob.glob(os.path.join(path, "*.parquet")) + glob.glob(os.path.join(path, "*.csv")) + glob.glob(os.path.join(path, "*.arff")))
# #         if not files:
# #             raise FileNotFoundError(f"Không thấy file hợp lệ trong thư mục: {path}")
# #         dfs = [read_file(f) for f in files]
# #         print(f"[INFO] Đã gộp {len(dfs)} file từ: {path}")
# #         return pd.concat(dfs, ignore_index=True)
# #     return read_file(path)

# # def infer_label_col(df: pd.DataFrame) -> str:
# #     for c in LABEL_CANDS:
# #         if c in df.columns:
# #             return c
# #     return df.columns[-1]

# # def infer_time_col(df: pd.DataFrame) -> str | None:
# #     for c in TIME_CANDS:
# #         if c in df.columns:
# #             return c
# #     return None

# # def coerce_numeric(df: pd.DataFrame, drop_cols: list[str]) -> pd.DataFrame:
# #     keep_cols = [c for c in df.columns if c not in drop_cols]
# #     out = df[keep_cols].copy()
# #     for c in out.columns:
# #         if not (pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c])):
# #             out[c] = pd.to_numeric(out[c], errors="ignore")
# #             if not (pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c])):
# #                 out[c] = out[c].astype("category").cat.codes

# #     out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
# #     return out

# # def normalize_labels(df: pd.DataFrame, label_col: str, binary: bool) -> tuple[pd.Series, list[str]]:
# #     y = df[label_col].astype(str).str.strip()
# #     if binary:
# #         def to_bin(s: str) -> str:
# #             s_low = s.lower()
# #             if s_low in ["benign","normal","0","non-attack","normal traffic","benign traffic"]:
# #                 return "Benign"
# #             if s_low.isdigit() and s_low in ["0","1"]:
# #                 return "Benign" if s_low=="0" else "Attack"
# #             return "Benign" if s_low in ["normal","norm"] else "Attack"
# #         y = y.apply(to_bin)
# #     le = LabelEncoder()
# #     y_enc = le.fit_transform(y)
# #     return pd.Series(y_enc, index=df.index), list(le.classes_)

# # def time_aware_split(df: pd.DataFrame, label_col: str, time_col: str|None, test_size=0.2, seed=42):
# #     if time_col and time_col in df.columns:
# #         tmp = df.copy()
# #         tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
# #         tmp = tmp.sort_values(time_col)
# #         n = len(tmp)
# #         split_idx = int((1-test_size)*n)
# #         tr, te = tmp.iloc[:split_idx], tmp.iloc[split_idx:]
# #         Xtr = coerce_numeric(tr.drop(columns=[label_col, time_col]), [])
# #         Xte = coerce_numeric(te.drop(columns=[label_col, time_col]), [])
# #         ytr = tr[label_col]; yte = te[label_col]
# #         return Xtr, Xte, ytr, yte
# #     X = coerce_numeric(df.drop(columns=[label_col]), [])
# #     y = df[label_col]
# #     return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

# # def drop_constant_cols(X: pd.DataFrame):
# #     nunique = X.nunique()
# #     keep = nunique[nunique > 1].index.tolist()
# #     removed = [c for c in X.columns if c not in keep]
# #     return X[keep], removed

# # def drop_corr_cols(X: pd.DataFrame, threshold=0.98):
# #     corr = X.corr(numeric_only=True).abs()
# #     upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
# #     to_drop = [c for c in upper.columns if any(upper[c] > threshold)]
# #     return X.drop(columns=to_drop), to_drop

# # def mi_topk(X: pd.DataFrame, y: np.ndarray, k: int):
# #     if k <= 0 or k >= X.shape[1]:
# #         return X, []
# #     mi = mutual_info_classif(X, y, random_state=42, discrete_features='auto')
# #     order = np.argsort(mi)[::-1][:k]
# #     cols = X.columns[order]
# #     removed = [c for c in X.columns if c not in cols]
# #     return X.loc[:, cols], removed

# # def measure_latency(model, X: pd.DataFrame, repeat=5, warmup=1000):
# #     n = min(warmup, len(X))
# #     if n > 0:
# #         _ = model.predict(X.iloc[:n])
# #     timings = []
# #     for _ in range(repeat):
# #         t0 = time.perf_counter()
# #         _ = model.predict(X)
# #         t1 = time.perf_counter()
# #         timings.append(t1 - t0)
# #     batch_s = float(np.median(timings)) if timings else 0.0
# #     per_sample_ms = (batch_s / max(1,len(X))) * 1000.0
# #     n2 = min(5000, len(X))
# #     t0 = time.perf_counter()
# #     for i in range(n2):
# #         _ = model.predict(X.iloc[i:i+1])
# #     t1 = time.perf_counter()
# #     single_ms = ((t1 - t0) / max(1,n2)) * 1000.0
# #     return per_sample_ms, single_ms, batch_s

# # def main():
# #     ap = argparse.ArgumentParser("IDS Benchmark V2 (FS + robust IO)")
# #     ap.add_argument("--input", required=True, help="File .parquet/.csv/.arff hoặc thư mục")
# #     ap.add_argument("--label-col", default=None, help="Tên cột nhãn (nếu muốn chỉ định)")
# #     ap.add_argument("--time-col", default=None, help="Tên cột thời gian (nếu có)")
# #     ap.add_argument("--binary", action="store_true", help="Benign vs Attack")
# #     ap.add_argument("--test-size", type=float, default=0.2)
# #     ap.add_argument("--seed", type=int, default=42)
# #     ap.add_argument("--max-rows", type=int, default=0)
# #     ap.add_argument("--drop-constant", action="store_true")
# #     ap.add_argument("--drop-corr", type=float, default=0.0, help="ngưỡng |corr| để loại (vd 0.98). 0 = bỏ qua")
# #     ap.add_argument("--mi-topk", type=int, default=0, help="chọn K đặc trưng MI cao nhất (0 = bỏ qua)")
# #     ap.add_argument("--save", default=None, help="Lưu kết quả JSON")
# #     args = ap.parse_args()

# #     df = read_any(args.input)
# #     if args.max_rows and len(df) > args.max_rows:
# #         df = df.sample(args.max_rows, random_state=args.seed).reset_index(drop=True)

# #     label_col = args.label_col or infer_label_col(df)
# #     time_col  = args.time_col  or infer_time_col(df)

# #     y_enc, class_names = normalize_labels(df, label_col, binary=args.binary)
# #     df = df.copy()
# #     df[label_col] = y_enc

# #     Xtr, Xte, ytr, yte = time_aware_split(df, label_col, time_col, test_size=args.test_size, seed=args.seed)

# #     removed_all = []

# #     if args.drop_constant:
# #         Xtr, rm = drop_constant_cols(Xtr); removed_all += rm

# #         Xte = Xte[[c for c in Xtr.columns]]

# #     if args.drop_corr and args.drop_corr > 0:
# #         Xtr, rm = drop_corr_cols(Xtr, threshold=args.drop_corr); removed_all += rm
# #         Xte = Xte[[c for c in Xtr.columns]]

# #     if args.mi_topk and args.mi_topk > 0:
# #         Xtr, rm = mi_topk(Xtr, ytr.values, k=args.mi_topk); removed_all += rm
# #         Xte = Xte[[c for c in Xtr.columns]]

# #     num_classes = len(np.unique(ytr))
# #     params = dict(
# #         objective="multiclass" if num_classes>2 else "binary",
# #         num_class=num_classes if num_classes>2 else None,
# #         learning_rate=0.08,
# #         num_leaves=63,
# #         n_estimators=240,
# #         n_jobs=-1,
# #         verbose=-1
# #     )
# #     model = lgb.LGBMClassifier(**{k:v for k,v in params.items() if v is not None})

# #     t0 = time.perf_counter()
# #     model.fit(Xtr, ytr)
# #     train_s = time.perf_counter() - t0

# #     ypred = model.predict(Xte)
# #     acc = accuracy_score(yte, ypred)
# #     f1m = f1_score(yte, ypred, average="macro")

# #     print(f"\n✅ Dataset: {args.input}")
# #     print(f"Rows: {len(df):,} | Label: {label_col} | Time col: {time_col if time_col else '(none)'}")
# #     print(f"Features dùng: {Xtr.shape[1]}  | Đã loại: {len(removed_all)}")
# #     print(f"Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
# #     print("-- Confusion Matrix --")
# #     print(confusion_matrix(yte, ypred))
# #     print("-- Classification Report --")
# #     try:
# #         print(classification_report(yte, ypred, target_names=class_names))
# #     except:
# #         print(classification_report(yte, ypred))

# #     per_ms, single_ms, batch_s = measure_latency(model, Xte)
# #     print(f"\n⚡ Train: {train_s:.2f}s | Batch pred: {batch_s:.3f}s | Latency: {per_ms:.3f} ms/flow | Single: {single_ms:.3f} ms/flow")

# #     if args.save:
# #         out = dict(
# #             path=args.input, rows=len(df), label_col=label_col, classes=class_names,
# #             time_col=time_col, features=Xtr.shape[1], removed_features=len(removed_all),
# #             accuracy=acc, macro_f1=f1m, train_time_s=train_s,
# #             batch_predict_s=batch_s, latency_ms_per_sample_batch=per_ms, latency_ms_single=single_ms
# #         )
# #         with open(args.save, "w", encoding="utf-8") as f:
# #             json.dump(out, f, ensure_ascii=False, indent=2)
# #         print(f"[OK] Lưu: {args.save}")

# # if __name__ == "__main__":
# #     main()


# # ids_cic2017_pipeline.py
# import os, glob, io, time, json
# import numpy as np
# import pandas as pd
# from typing import Optional, Tuple, List, Dict

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_selection import mutual_info_classif
# from sklearn.base import BaseEstimator, TransformerMixin

# import lightgbm as lgb

# # ====== Cấu hình ======
# LABEL_CANDS = ["Label","label","Attack","attack","attack_cat","class","Class","target","category","Category","result"]
# TIME_CANDS  = ["Timestamp","timestamp","Flow ID","FlowID","StartTime","Start Time","stime","time","Date","datetime"]

# # ====== Đọc CSV an toàn ======
# def safe_read_csv(path: str) -> pd.DataFrame:
#     encodings = ["utf-8-sig","utf-8","cp1252","latin1"]
#     for enc in encodings:
#         try:
#             return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
#         except Exception:
#             continue
#     with open(path, "rb") as f:
#         data = f.read().decode("latin1", errors="ignore")
#     return pd.read_csv(io.StringIO(data), sep=None, engine="python", on_bad_lines="skip")

# # ====== Đọc file (đã vá lỗi định dạng) ======
# def read_file(f: str) -> pd.DataFrame:
#     f = os.path.normpath(f)
#     ext = os.path.splitext(f)[1].lower()

#     if ext in (".csv", ".txt", ".data"):
#         return safe_read_csv(f)
#     if f.lower().endswith(".csv.gz"):
#         return pd.read_csv(f, compression="gzip", sep=None, engine="python", on_bad_lines="skip")
#     if ext == ".parquet":
#         return pd.read_parquet(f)
#     if ext in (".xlsx", ".xls"):
#         return pd.read_excel(f)
#     if ext == ".arff":
#         import liac_arff  # type: ignore
#         with open(f, 'r', errors="ignore") as fp:
#             arff = liac_arff.load(fp)
#         return pd.DataFrame(arff['data'], columns=[c[0] for c in arff['attributes']])
#     if ext == "":
#         try:
#             return safe_read_csv(f)
#         except Exception:
#             pass
#     raise ValueError(f"Định dạng chưa hỗ trợ hoặc không nhận diện được: '{ext}' cho file: {f}")

# # ====== Đọc thư mục hoặc file ======
# def read_any(path: str) -> pd.DataFrame:
#     path = os.path.normpath(str(path)).strip().strip('"').strip("'")

#     if os.path.isdir(path):
#         patterns = ["*.parquet","*.csv","*.arff","*.xlsx","*.xls","*.txt","*.data","*.csv.gz"]
#         files: list[str] = []
#         for pat in patterns:
#             files += glob.glob(os.path.join(path, "**", pat), recursive=True)

#         if not files:
#             raise FileNotFoundError(
#                 f"Không tìm thấy file hợp lệ trong thư mục: {path}\n"
#                 f"Cần có đuôi: {', '.join(patterns)}"
#             )
#         print(f"[INFO] Đã gộp {len(files)} file từ: {path}")
#         dfs = [read_file(f) for f in sorted(files)]
#         return pd.concat(dfs, ignore_index=True)

#     if os.path.isfile(path):
#         return read_file(path)

#     # fallback nếu path không rõ
#     patterns = ["*.parquet","*.csv","*.arff","*.xlsx","*.xls","*.txt","*.data","*.csv.gz"]
#     files = []
#     for pat in patterns:
#         files += glob.glob(os.path.join(path, "**", pat), recursive=True)
#     if files:
#         print(f"[WARN] '{path}' không nhận diện được bằng os.path.isdir, nhưng tìm thấy {len(files)} file.")
#         dfs = [read_file(f) for f in sorted(files)]
#         return pd.concat(dfs, ignore_index=True)

#     raise FileNotFoundError(
#         f"Không tìm thấy đường dẫn hoặc file hợp lệ.\n"
#         f"Đường dẫn: {path}\n"
#         "Kiểm tra: đường dẫn tuyệt đối, bỏ dấu ngoặc kép thừa, thư mục chứa file .csv/.parquet/.arff..."
#     )

# # ====== Suy đoán cột ======
# def infer_label_col(df: pd.DataFrame) -> str:
#     for c in LABEL_CANDS:
#         if c in df.columns: return c
#     return df.columns[-1]

# def infer_time_col(df: pd.DataFrame) -> Optional[str]:
#     for c in TIME_CANDS:
#         if c in df.columns: return c
#     return None

# # ====== Chuẩn hóa nhãn CIC-IDS2017 ======
# def normalize_cic2017_labels(y_raw: pd.Series) -> Tuple[pd.Series, List[str]]:
#     def norm(s: str) -> str:
#         s = str(s).strip()
#         s_up = s.upper()
#         if s_up in ["BENIGN","BENIGN TRAFFIC","NORMAL","NON-ATTACK","0"]:
#             return "BENIGN"
#         if "WEB ATTACK" in s_up:
#             s_up = s_up.replace("–","-").replace("—","-")
#             if "BRUTE" in s_up: return "WEB ATTACK - BRUTE FORCE"
#             if "XSS" in s_up: return "WEB ATTACK - XSS"
#             if "SQL" in s_up: return "WEB ATTACK - SQL INJECTION"
#             return "WEB ATTACK"
#         for k in ["DDOS","PORTSCAN","BOT","INFILTRATION","HEARTBLEED"]:
#             if k in s_up: return k
#         return s_up
#     y_norm = y_raw.astype(str).apply(norm)
#     le = LabelEncoder()
#     y_enc = le.fit_transform(y_norm)
#     return pd.Series(y_enc, index=y_raw.index), list(le.classes_)

# # ====== Ép numeric nhanh ======
# def coerce_numeric(df: pd.DataFrame, drop_cols: List[str]) -> pd.DataFrame:
#     keep_cols = [c for c in df.columns if c not in drop_cols]
#     out = df[keep_cols].copy()
#     for c in out.columns:
#         if not (pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c])):
#             out[c] = pd.to_numeric(out[c], errors="ignore")
#             if not (pd.api.types.is_integer_dtype(out[c]) or pd.api.types.is_float_dtype(out[c])):
#                 out[c] = out[c].astype("category").cat.codes
#     out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
#     return out

# # ====== Chia train/test 8:2 ======
# def split_80_20(df: pd.DataFrame, label_col: str, time_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     if time_col and time_col in df.columns:
#         tmp = df.copy()
#         tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
#         tmp = tmp.sort_values(time_col)
#         n = len(tmp)
#         split_idx = int(0.8 * n)
#         tr, te = tmp.iloc[:split_idx], tmp.iloc[split_idx:]
#         Xtr = coerce_numeric(tr.drop(columns=[label_col, time_col]), [])
#         Xte = coerce_numeric(te.drop(columns=[label_col, time_col]), [])
#         ytr = tr[label_col]; yte = te[label_col]
#         return Xtr, Xte, ytr, yte
#     X = coerce_numeric(df.drop(columns=[label_col]), [])
#     y = df[label_col]
#     return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# # ====== Bộ lọc đặc trưng ======
# class ConstantFilter(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         X = pd.DataFrame(X)
#         nunique = X.nunique()
#         self.keep_cols_ = nunique[nunique > 1].index.tolist()
#         return self
#     def transform(self, X):
#         return pd.DataFrame(X)[self.keep_cols_]

# class CorrelationFilter(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold: float = 0.98):
#         self.threshold = threshold
#     def fit(self, X, y=None):
#         X = pd.DataFrame(X)
#         corr = X.corr(numeric_only=True).abs()
#         upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
#         to_drop = set([c for c in upper.columns if any(upper[c] > self.threshold)])
#         self.keep_cols_ = [c for c in X.columns if c not in to_drop]
#         return self
#     def transform(self, X):
#         return pd.DataFrame(X)[self.keep_cols_]

# class MITopK(BaseEstimator, TransformerMixin):
#     def __init__(self, k: int = 0, random_state: int = 42):
#         self.k = k; self.random_state = random_state
#     def fit(self, X, y):
#         X = pd.DataFrame(X)
#         if self.k <= 0 or self.k >= X.shape[1]:
#             self.keep_cols_ = list(X.columns)
#             return self
#         mi = mutual_info_classif(X.values, np.asarray(y), random_state=self.random_state, discrete_features='auto')
#         order = np.argsort(mi)[::-1][:self.k]
#         self.keep_cols_ = X.columns[order].tolist()
#         return self
#     def transform(self, X):
#         return pd.DataFrame(X)[self.keep_cols_]

# # ====== Pipeline chính ======
# def run_cic2017_pipeline(
#     input_path: str,
#     max_rows: int = 0,
#     corr_threshold: float = 0.98,
#     mi_topk: int = 0,
#     lgb_params: Dict = None,
#     save_json: Optional[str] = None
# ) -> Dict:
#     df = read_any(input_path)
#     if max_rows and len(df) > max_rows:
#         df = df.sample(max_rows, random_state=42).reset_index(drop=True)

#     label_col = infer_label_col(df)
#     time_col  = infer_time_col(df)

#     y_enc, class_names = normalize_cic2017_labels(df[label_col])
#     df[label_col] = y_enc

#     Xtr, Xte, ytr, yte = split_80_20(df, label_col, time_col)

#     # Lọc đặc trưng
#     Xtr = ConstantFilter().fit_transform(Xtr)
#     Xte = Xte[Xtr.columns]
#     if corr_threshold > 0:
#         Xtr = CorrelationFilter(threshold=corr_threshold).fit_transform(Xtr)
#         Xte = Xte[Xtr.columns]
#     if mi_topk > 0:
#         Xtr = MITopK(k=mi_topk).fit_transform(Xtr, ytr)
#         Xte = Xte[Xtr.columns]

#     # LightGBM
#     num_classes = len(np.unique(ytr))
#     params = dict(
#         objective="multiclass" if num_classes>2 else "binary",
#         num_class=num_classes if num_classes>2 else None,
#         learning_rate=0.08,
#         num_leaves=63,
#         n_estimators=240,
#         n_jobs=-1,
#         verbose=-1
#     )
#     if lgb_params: params.update(lgb_params)
#     model = lgb.LGBMClassifier(**{k:v for k,v in params.items() if v is not None})

#     t0 = time.perf_counter()
#     model.fit(Xtr, ytr)
#     train_s = time.perf_counter() - t0

#     ypred = model.predict(Xte)
#     acc  = accuracy_score(yte, ypred)
#     f1m  = f1_score(yte, ypred, average="macro")

#     cm = confusion_matrix(yte, ypred)
#     try:
#         report = classification_report(yte, ypred, target_names=class_names)
#     except:
#         report = classification_report(yte, ypred)

#     print(f"\n✅ Dataset: {input_path}")
#     print(f"Rows: {len(df):,} | Label: {label_col} | Time col: {time_col or '(none)'}")
#     print(f"Lớp: {class_names}")
#     print(f"Features: {Xtr.shape[1]}")
#     print(f"Accuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
#     print("-- Confusion Matrix --")
#     print(cm)
#     print("-- Classification Report --")
#     print(report)

#     out = dict(
#         path=input_path, rows=len(df),
#         label_col=label_col, classes=class_names,
#         accuracy=acc, macro_f1=f1m,
#         confusion_matrix=cm.tolist(),
#         classification_report=report,
#         train_time_s=train_s, features=Xtr.shape[1]
#     )
#     if save_json:
#         with open(save_json, "w", encoding="utf-8") as f:
#             json.dump(out, f, ensure_ascii=False, indent=2)
#         print(f"[OK] Lưu: {save_json}")
#     return out

# if __name__ == "__main__":
#     INPUT_PATH = r"D:\DACN\dataset\CICDDoS2017"  # <-- chỉnh đúng đường dẫn thư mục hoặc file
#     run_cic2017_pipeline(
#         input_path=INPUT_PATH,
#         max_rows=0,
#         corr_threshold=0.98,
#         mi_topk=0,
#         save_json="results_cic2017_multiclass.json"
#     )


# ids_cic2017_epochs_tf.py
# Train theo EPOCH (Keras) + LR scheduler + EarlyStopping
# Union schema (không bỏ sót cột), 8/2 split, class_weight balanced
import os, glob, io, time, json, argparse
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

LABEL_CANDS = ["Label","label","Attack","attack","attack_cat","class","Class","target","category","Category","result"]
TIME_CANDS  = ["Timestamp","timestamp","Flow ID","FlowID","StartTime","Start Time","stime","time","Date","datetime"]

def safe_read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig","utf-8","cp1252","latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc, on_bad_lines="skip")
        except Exception:
            pass
    with open(path, "rb") as f:
        data = f.read().decode("latin1", errors="ignore")
    return pd.read_csv(io.StringIO(data), sep=None, engine="python", on_bad_lines="skip")

def read_file(f: str) -> pd.DataFrame:
    f = os.path.normpath(f)
    ext = os.path.splitext(f)[1].lower()
    if ext in (".csv", ".txt", ".data"): return safe_read_csv(f)
    if f.lower().endswith(".csv.gz"):    return pd.read_csv(f, compression="gzip", sep=None, engine="python", on_bad_lines="skip")
    if ext == ".parquet":                return pd.read_parquet(f)
    if ext in (".xlsx", ".xls"):         return pd.read_excel(f)
    if ext == ".arff":
        import liac_arff  # type: ignore
        with open(f, 'r', errors="ignore") as fp:
            arff = liac_arff.load(fp)
        return pd.DataFrame(arff['data'], columns=[c[0] for c in arff['attributes']])
    if ext == "":
        try:    return safe_read_csv(f)
        except: pass
    raise ValueError(f"Định dạng không hỗ trợ: {f}")

def infer_label_col(df: pd.DataFrame) -> str:
    for c in LABEL_CANDS:
        if c in df.columns: return c
    return df.columns[-1]

def infer_time_col(df: pd.DataFrame) -> Optional[str]:
    for c in TIME_CANDS:
        if c in df.columns: return c
    return None

def normalize_cic2017_label_text(s: str) -> str:
    u = str(s).strip().upper()
    if u in ["BENIGN","BENIGN TRAFFIC","NORMAL","NON-ATTACK","0"]: return "BENIGN"
    if "WEB ATTACK" in u:
        u = u.replace("–","-").replace("—","-")
        if "BRUTE" in u: return "WEB ATTACK - BRUTE FORCE"
        if "XSS"   in u: return "WEB ATTACK - XSS"
        if "SQL"   in u: return "WEB ATTACK - SQL INJECTION"
        return "WEB ATTACK"
    for k in ["DDOS","DOS HULK","DOS SLOWHTTPTEST","DOS SLOWLORIS","DOS GOLDENEYE",
              "FTP-PATATOR","SSH-PATATOR","BOT","PORTSCAN","INFILTRATION","HEARTBLEED"]:
        if k in u: return k
    return u

def to_numeric_keep_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]): continue
        tmp = pd.to_numeric(out[c], errors="coerce")
        if tmp.notna().sum() >= 0.7 * len(tmp):
            out[c] = tmp
        else:
            out[c] = out[c].astype("category").cat.codes
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
    return out

def load_cic2017_union_schema(input_dir: str) -> Tuple[pd.DataFrame, str, Optional[str], List[str], List[str]]:
    input_dir = os.path.normpath(input_dir).strip().strip('"').strip("'")
    patterns = ["*.csv","*.csv.gz","*.parquet","*.xlsx","*.xls","*.txt","*.data"]
    files: List[str] = []
    for p in patterns:
        files += glob.glob(os.path.join(input_dir, "**", p), recursive=True)
    if not files: raise FileNotFoundError(f"Không tìm thấy file trong: {input_dir}")

    all_cols: Set[str] = set()
    all_labels: Set[str] = set()
    label_col_name: Optional[str] = None
    time_col_name: Optional[str]  = None

    print("\n[INFO] Quét file & nhãn:")
    for f in sorted(files):
        df = read_file(f)
        lbl = infer_label_col(df); tmc = infer_time_col(df)
        if label_col_name is None: label_col_name = lbl
        if time_col_name  is None and tmc is not None: time_col_name = tmc
        all_cols.update(df.columns.tolist())
        vals = df[lbl].astype(str).unique().tolist()
        nlabels = sorted(set(normalize_cic2017_label_text(v) for v in vals))
        all_labels.update(nlabels)
        print(f" - {os.path.basename(f)} | rows={len(df)} | cols={df.shape[1]} | labels={nlabels}")

    union_cols = list(all_cols)
    big_parts: List[pd.DataFrame] = []
    for f in sorted(files):
        df = read_file(f)
        missing = [c for c in union_cols if c not in df.columns]
        for c in missing: df[c] = 0
        big_parts.append(df[union_cols])

    big = pd.concat(big_parts, ignore_index=True)
    class_list = sorted(all_labels)
    print(f"\n[INFO] Union cột = {len(union_cols)}, lớp = {class_list}")
    return big, (label_col_name or union_cols[-1]), time_col_name, class_list, union_cols

def split_80_20(df: pd.DataFrame, label_col: str, time_col: Optional[str]):
    if time_col and time_col in df.columns:
        tmp = df.copy()
        tmp[time_col] = pd.to_datetime(tmp[time_col], errors="coerce")
        tmp = tmp.sort_values(time_col)
        n = len(tmp); cut = int(0.8*n)
        tr, te = tmp.iloc[:cut], tmp.iloc[cut:]
        Xtr = tr.drop(columns=[label_col, time_col]); ytr = tr[label_col]
        Xte = te.drop(columns=[label_col, time_col]); yte = te[label_col]
    else:
        X = df.drop(columns=[label_col]); y = df[label_col]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return Xtr, Xte, ytr, yte

def save_confusion_matrices(cm: np.ndarray, classes: List[str], outdir: str):
    df = pd.DataFrame(cm, index=[f"true:{c}" for c in classes], columns=[f"pred:{c}" for c in classes])
    df.to_csv(os.path.join(outdir, "confusion_matrix.csv"), encoding="utf-8")
    # raw
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thresh = cm.max()/2. if cm.size else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm[i,j])), ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_matrix_raw.png"), bbox_inches="tight"); plt.close(fig)
    # normalized
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.maximum(row_sum,1), where=(row_sum!=0))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    ax.set_title("Confusion Matrix (normalized)"); ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    thresh = norm.max()/2. if norm.size else 0.5
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if norm[i,j] > thresh else "black")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "confusion_matrix_normalized.png"), bbox_inches="tight"); plt.close(fig)

def main():
    ap = argparse.ArgumentParser("CIC-IDS2017 Epoch Trainer (Keras)")
    ap.add_argument("--input", required=True, help="Thư mục CIC-IDS2017")
    ap.add_argument("--outdir", default="results_cic2017_tf", help="Nơi lưu kết quả")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--hidden", type=int, nargs="*", default=[512, 256, 128], help="Các tầng ẩn Dense")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    raw, label_col, time_col, class_list, union_cols = load_cic2017_union_schema(args.input)

    # 2) Nhãn
    y_text = raw[label_col].astype(str).map(normalize_cic2017_label_text)
    le = LabelEncoder(); y = le.fit_transform(y_text)
    classes = le.classes_.tolist(); num_classes = len(classes)

    drop_cols = [label_col] + ([time_col] if time_col and time_col in raw.columns else [])
    X_all = to_numeric_keep_all(raw.drop(columns=drop_cols)).astype(np.float32)

    # 4) Split 8/2
    df = X_all.copy()
    df[label_col] = y
    if time_col and time_col in raw.columns:
        df[time_col] = raw[time_col]
    Xtr, Xte, ytr, yte = split_80_20(df, label_col, time_col)
    Xte = Xte[Xtr.columns]  # đồng bộ cột

    # 5) Scale theo train để NN hội tụ tốt
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xtr_np = scaler.fit_transform(Xtr.values).astype(np.float32)
    Xte_np = scaler.transform(Xte.values).astype(np.float32)

    # 6) One-hot y
    ytr_oh = tf.keras.utils.to_categorical(ytr, num_classes=num_classes)
    yte_oh = tf.keras.utils.to_categorical(yte, num_classes=num_classes)

    # 7) Class weights để xử lý mất cân bằng
    cls_w = compute_class_weight(class_weight="balanced", classes=np.unique(ytr), y=ytr)
    class_weight = {int(k): float(v) for k, v in zip(np.unique(ytr), cls_w)}

    # 8) Xây model Dense
    inputs = tf.keras.Input(shape=(Xtr_np.shape[1],), name="features")
    x = inputs
    for h in args.hidden:
        x = tf.keras.layers.Dense(h, activation=None)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 9) Callbacks giống log bạn chụp
    ckpt_path = os.path.join(args.outdir, "best_model.keras")
    cbs = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy", factor=0.5, patience=2,
            min_lr=3e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6,
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=1)
    ]

    # 10) Train — sẽ in log từng Epoch
    history = model.fit(
        Xtr_np, ytr_oh,
        validation_data=(Xte_np, yte_oh),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        verbose=1,
        callbacks=cbs
    )

    # 11) Đánh giá + CM
    ypred_probs = model.predict(Xte_np, batch_size=args.batch_size, verbose=0)
    ypred = np.argmax(ypred_probs, axis=1)
    acc  = accuracy_score(yte, ypred)
    f1m  = f1_score(yte, ypred, average="macro")
    cm   = confusion_matrix(yte, ypred)

    print(f"\nAccuracy: {acc:.4f} | F1-macro: {f1m:.4f}")
    try:
        print(classification_report(yte, ypred, target_names=classes))
    except:
        print(classification_report(yte, ypred))

    # 12) Lưu CM + lịch sử học (hình)
    save_confusion_matrices(cm, classes, args.outdir)

    # loss/acc curve
    hist = pd.DataFrame(history.history)
    hist.to_csv(os.path.join(args.outdir, "history.csv"), index=False, encoding="utf-8")
    plt.figure(figsize=(8,4)); plt.plot(hist["loss"], label="loss"); plt.plot(hist["val_loss"], label="val_loss"); plt.legend(); plt.title("Loss")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "loss_curve.png")); plt.close()
    plt.figure(figsize=(8,4)); plt.plot(hist["accuracy"], label="accuracy"); plt.plot(hist["val_accuracy"], label="val_accuracy"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout(); plt.savefig(os.path.join(args.outdir, "acc_curve.png")); plt.close()

    # 13) JSON summary
    summary = dict(
        dataset=args.input, rows=int(len(raw)),
        label_col=label_col, time_col=time_col,
        n_features=int(Xtr.shape[1]), classes=classes,
        accuracy=float(acc), f1_macro=float(f1m),
        artifacts=dict(
            best_model=ckpt_path,
            cm_csv=os.path.join(args.outdir, "confusion_matrix.csv"),
            cm_png=os.path.join(args.outdir, "confusion_matrix_raw.png"),
            cm_norm_png=os.path.join(args.outdir, "confusion_matrix_normalized.png"),
            history_csv=os.path.join(args.outdir, "history.csv"),
            loss_curve=os.path.join(args.outdir, "loss_curve.png"),
            acc_curve=os.path.join(args.outdir, "acc_curve.png")
        )
    )
    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] Saved to: {args.outdir}")

if __name__ == "__main__":
    main()
