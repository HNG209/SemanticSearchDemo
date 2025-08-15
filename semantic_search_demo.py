import os
import hashlib
import torch
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility
from sentence_transformers import SentenceTransformer

# ===================== CONFIG =====================
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "image_collection"
IMAGES_DIR = "images"
DIM = 512  # CLIP ViT-B-16 embedding size
TOP_K = 8
PAGE_SIZE_DEFAULT = 50

print(torch.cuda.get_device_name(0))
print('CUDA available:', torch.cuda.is_available())  # True nếu GPU sẵn sàng
os.makedirs(IMAGES_DIR, exist_ok=True)

# ===================== MILVUS CONNECT =====================
try:
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
except Exception as e:
    print(f"[Milvus] Connect error: {e}")


def init_collection():
    """Create collection and index if not exists, return Collection object."""
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
            FieldSchema(name="filepath", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="image_vector", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        ]
        schema = CollectionSchema(fields, description="Image semantic search (CLIP)")
        col = Collection(COLLECTION_NAME, schema)
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        col.create_index(field_name="image_vector", index_params=index_params)
    else:
        col = Collection(COLLECTION_NAME)
    return col


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
    """Insert a single image file into Milvus and save a copy to images/.
    Returns (pk, filename)."""
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

    col = init_collection()
    data = [[pk_value], [filename], [vec]]
    col.insert(data)
    col.flush()
    return pk_value, filename


def search_by_text(query: str, top_k: int = TOP_K):
    col = init_collection()
    col.load()
    qvec = embed_text(query)
    params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    res = col.search(
        data=[qvec],
        anns_field="image_vector",
        param=params,
        limit=top_k,
        output_fields=["filepath", "pk"],
    )
    return res[0]


def fetch_page(page: int, page_size: int):
    """Fetch a page of rows. Milvus doesn't support offset; emulate with large limit and slicing."""
    col = init_collection()
    col.load()
    # Use a conservative cap; for big data, implement server-side filters instead
    limit = min((page + 1) * page_size, 10000)
    rows = col.query(expr="", output_fields=["pk", "filepath"], limit=limit)
    start = page * page_size
    return rows[start:start + page_size]


def delete_row(pk_value: str):
    col = init_collection()
    # Với VARCHAR phải đặt trong ngoặc kép và dùng IN
    if isinstance(pk_value, str):
        expr = f'pk in ["{pk_value}"]'
    else:
        expr = f"pk in [{pk_value}]"
    col.delete(expr=expr)
    col.flush()


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
notebook.add(tab_upload, text="📤 Upload ảnh")

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
                fail += 1
        status_upload.config(text=f"Đã chèn {ok} ảnh, lỗi {fail} ảnh.")
        refresh_table()

    threading.Thread(target=_worker, args=(file_paths,), daemon=True).start()


btn_upload = ttk.Button(tab_upload, text="Chọn ảnh & chèn vào Milvus", command=do_upload)
btn_upload.pack(pady=6)

# ---------- TAB 2: SEARCH (TEXT -> IMAGE) ----------
tab_search = ttk.Frame(notebook)
notebook.add(tab_search, text="🔎 Tìm bằng văn bản")

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
        fp = h.entity.get("filepath")
        score = getattr(h, 'score', getattr(h, 'distance', 0.0))
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

# ---------- TAB 3: VIEW DATA (TABLE + DELETE PER ROW) ----------
tab_view = ttk.Frame(notebook)
notebook.add(tab_view, text="📚 Xem dữ liệu & Xoá")

frm_top = ttk.Frame(tab_view)
frm_top.pack(fill=tk.X, padx=10, pady=6)

page_var = tk.IntVar(value=0)
page_size_var = tk.IntVar(value=PAGE_SIZE_DEFAULT)


def refresh_table():
    try:
        page = page_var.get()
        size = page_size_var.get()
        rows = fetch_page(page, size)
        tree.delete(*tree.get_children())
        for r in rows:
            tree.insert("", tk.END, values=(r.get("pk"), r.get("filepath"), "🗑️ Xoá"))
        lbl_page.config(text=f"Trang {page + 1}")
    except Exception as e:
        messagebox.showerror("Lỗi tải dữ liệu", str(e))


btn_prev = ttk.Button(frm_top, text="◀ Trước", command=lambda: (page_var.set(max(0, page_var.get()-1)), refresh_table()))
btn_prev.pack(side=tk.LEFT)

lbl_page = ttk.Label(frm_top, text="Trang 1")
lbl_page.pack(side=tk.LEFT, padx=8)

btn_next = ttk.Button(frm_top, text="Tiếp ▶", command=lambda: (page_var.set(page_var.get()+1), refresh_table()))
btn_next.pack(side=tk.LEFT)

ttk.Label(frm_top, text="Kích thước trang:").pack(side=tk.LEFT, padx=10)
spin_size = ttk.Spinbox(frm_top, from_=10, to=500, textvariable=page_size_var, width=6, command=refresh_table)
spin_size.pack(side=tk.LEFT)

cols = ("pk", "filepath", "action")

container = ttk.Frame(tab_view)
container.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

tree = ttk.Treeview(container, columns=cols, show="headings")
for c, w in zip(cols, (280, 520, 80)):
    tree.heading(c, text=c.upper())
    tree.column(c, width=w, anchor=tk.W)

vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
hsb = ttk.Scrollbar(container, orient="horizontal", command=tree.xview)

tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

tree.grid(row=0, column=0, sticky="nsew")
vsb.grid(row=0, column=1, sticky="ns")
hsb.grid(row=1, column=0, sticky="ew")
container.grid_rowconfigure(0, weight=1)
container.grid_columnconfigure(0, weight=1)

preview = tk.Label(tab_view)
preview.pack(pady=6)


def on_tree_click(event):
    item_id = tree.identify_row(event.y)
    col = tree.identify_column(event.x)
    if not item_id:
        return

    vals = tree.item(item_id).get("values", [])
    if len(vals) < 2:
        return

    pk_val, fp = vals[0], vals[1]
    img_path = os.path.join(IMAGES_DIR, os.path.basename(fp))

    # Show preview on row click (any column)
    if os.path.exists(img_path):
        try:
            im = Image.open(img_path)
            im.thumbnail((260, 260))
            imgtk = ImageTk.PhotoImage(im)
            preview.configure(image=imgtk)
            preview.image = imgtk
        except Exception:
            preview.configure(text="(Không hiển thị được ảnh)")
    else:
        preview.configure(text="(Thiếu file ảnh trên ổ đĩa)")

    # If click on action column -> delete
    if col == "#3":
        if messagebox.askyesno("Xoá", f"Xoá bản ghi pk={pk_val}?"):
            try:
                delete_row(pk_val)
                # Optionally remove local file copy
                if os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                    except Exception:
                        pass
                refresh_table()
            except Exception as e:
                messagebox.showerror("Lỗi xoá", str(e))


tree.bind("<Button-1>", on_tree_click)

# Initial table load
refresh_table()

# Start UI
root.mainloop()
