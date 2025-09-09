import os
import hashlib
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

import pinecone
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
PINECONE_API_KEY = "pcsk_4277or_KnoLJfyChRyDXZSdQU3i7sqcbcv4NQwdHvHaCvs93e5kGGpaSHKpzERmp9YrCM1"
PINECONE_ENV = "us-west1-gcp"
INDEX_NAME = "image-index"

IMAGES_DIR = "images"
# DIM = 512  # CLIP ViT-B-16 embedding size
TOP_K = 8
PAGE_SIZE_DEFAULT = 50

pc = Pinecone(api_key=PINECONE_API_KEY)

# Kết nối tới index
index = pc.Index(INDEX_NAME)

# ===================== MODEL (CLIP) =====================
# Use CPU to avoid GPU driver issues; trust_remote_code to avoid meta tensor lazy-load quirks
model = SentenceTransformer('clip-ViT-B-16', device="cuda", trust_remote_code=True)


# ===================== BACKEND HELPERS =====================

def hash_pk(filename: str) -> str:
    return hashlib.md5(filename.encode()).hexdigest()


def embed_image(pil_image: Image.Image):
    # Ensure RGB and CLIP input size
    img = pil_image.convert("RGB").resize((224, 224))
    return model.encode(img).tolist()


def embed_text(query: str):
    # CLIP can embed text to the same space as images
    return model.encode([query])[0].tolist()


def insert_image(file_path: str):
    filename = os.path.basename(file_path)
    pk_value = hash_pk(filename)

    # Save a local copy for previewing in UI
    dst_path = os.path.join(IMAGES_DIR, filename)
    try:
        Image.open(file_path).convert("RGB").save(dst_path)
    except Exception:
        # If cannot save (e.g., permission), continue with embedding
        pass

    # Build embedding
    vec = embed_image(Image.open(file_path))

    index.upsert([(pk_value, vec, {"filepath": filename})])
    return pk_value, filename


def search_by_text(query: str, top_k: int = TOP_K):
    qvec = embed_text(query)
    results = index.query(
        vector=qvec,
        top_k=top_k,
        namespace="",
        include_metadata=True
    )
    return results.matches


def fetch_page(page: int, page_size: int):
    # Pinecone không hỗ trợ offset, nên cần tự quản lý danh sách ID
    ids = list(index.describe_index_stats()["namespaces"].get("", {}).get("vector_count", 0))
    # cái này chỉ cho biết số lượng vector, muốn phân trang bạn cần lưu list pk_value trong file/local DB
    return []


def delete_row(pk_value: str):
    index.delete([pk_value])


# ===================== UI (TKINTER) =====================
root = tk.Tk()
root.title("Semantic Search Demo")
root.geometry("980x700")

style = ttk.Style()
style.configure("Treeview", rowheight=26)

notebook = ttk.Notebook(root)
notebook.pack(fill=tk.BOTH, expand=True)

# ---------- TAB 1: UPLOAD ----------
tab_upload = ttk.Frame(notebook)
notebook.add(tab_upload, text="Upload ảnh")

upload_preview = tk.Label(tab_upload)
upload_preview.pack(pady=8)

status_upload = tk.Label(tab_upload, text="", fg="green")
status_upload.pack(pady=4)


def do_upload():
    file_paths = filedialog.askopenfilenames(title="Chọn ảnh", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.webp")])
    if not file_paths:
        return

    def _worker(paths):
        ok, fail = 0, 0
        for p in paths:
            try:
                img = Image.open(p)
                # Show quick preview (first only)
                img_prev = img.copy()
                img_prev.thumbnail((320, 320))
                imgtk = ImageTk.PhotoImage(img_prev)
                upload_preview.configure(image=imgtk)
                upload_preview.image = imgtk

                insert_image(p)
                ok += 1
            except Exception as e:
                print(f"Error uploading {p}: {e}")
                fail += 1
        status_upload.config(text=f"Đã chèn {ok} ảnh, lỗi {fail} ảnh.")

    threading.Thread(target=_worker, args=(file_paths,), daemon=True).start()


btn_upload = ttk.Button(tab_upload, text="Chọn ảnh & chèn vào Pinecone", command=do_upload)
btn_upload.pack(pady=6)

# ---------- TAB 2: SEARCH (TEXT -> IMAGE) ----------
tab_search = ttk.Frame(notebook)
notebook.add(tab_search, text="Tìm bằng văn bản")

frm_q = ttk.Frame(tab_search)
frm_q.pack(fill=tk.X, padx=10, pady=8)

lbl_q = ttk.Label(frm_q, text="Nhập mô tả (prompt):")
lbl_q.pack(side=tk.LEFT)

entry_q = ttk.Entry(frm_q)
entry_q.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

res_panel = ttk.Frame(tab_search)
res_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

canvas = tk.Canvas(res_panel)
scroll_y = ttk.Scrollbar(res_panel, orient="vertical", command=canvas.yview)
inner = ttk.Frame(canvas)
inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
canvas.create_window((0, 0), window=inner, anchor="nw")
canvas.configure(yscrollcommand=scroll_y.set)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

search_status = tk.Label(tab_search, text="", fg="green")
search_status.pack(pady=4)

thumb_refs = []  # keep references to PhotoImage to avoid GC


def render_results(hits):
    for w in inner.winfo_children():
        w.destroy()
    thumb_refs.clear()

    if not hits:
        ttk.Label(inner, text="Không có kết quả.").pack(pady=8)
        return

    for h in hits:
        fp = h["metadata"]["filepath"]
        score = h["score"]
        path = os.path.join(IMAGES_DIR, os.path.basename(fp))
        frame = ttk.Frame(inner)
        frame.pack(padx=6, pady=6, anchor="w")

        if os.path.exists(path):
            try:
                img = Image.open(path)
                img.thumbnail((160, 160))
                imgtk = ImageTk.PhotoImage(img)
                thumb_refs.append(imgtk)
                tk.Label(frame, image=imgtk).pack(side=tk.LEFT, padx=6)
            except Exception:
                ttk.Label(frame, text="(Không hiển thị được ảnh)").pack(side=tk.LEFT, padx=6)
        else:
            ttk.Label(frame, text="(Thiếu file ảnh trên ổ đĩa)").pack(side=tk.LEFT, padx=6)

        meta = ttk.Frame(frame)
        meta.pack(side=tk.LEFT, padx=8)
        ttk.Label(meta, text=f"{fp}").pack(anchor="w")
        ttk.Label(meta, text=f"score: {score:.4f}").pack(anchor="w")


def do_search():
    q = entry_q.get().strip()
    if not q:
        messagebox.showwarning("Thiếu nội dung", "Vui lòng nhập mô tả tìm kiếm.")
        return

    def _worker():
        try:
            hits = search_by_text(q, top_k=TOP_K)
            search_status.config(text=f"Tìm thấy {len(hits)} kết quả")
            render_results(hits)
        except Exception as e:
            messagebox.showerror("Lỗi tìm kiếm", str(e))

    threading.Thread(target=_worker, daemon=True).start()


btn_search = ttk.Button(frm_q, text="Tìm", command=do_search)
btn_search.pack(side=tk.LEFT)

# Start UI
root.mainloop()
