import os
import sqlite3
import numpy as np
import faiss
from fastembed import TextEmbedding

# -------------------
# Embedding Manager
# -------------------
class EmbeddingManager:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        # loads only once, much faster than sentence-transformers
        self.model = TextEmbedding(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        # fastembed returns a generator, so convert to list
        return np.array(list(self.model.embed(texts)), dtype=np.float32)

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# -------------------
# Short Term Memory
# -------------------
class ShortTermMemory:
    def __init__(self):
        self.current_directory = os.getcwd()
        self.last_command = None
        self.last_output = None
        self.recent_folders = {}
        self.recent_files = {}

    def update(self, command, output):
        self.last_command = command
        self.last_output = output
        self.current_directory = os.getcwd()

    def remember_folder(self, name, path):
        self.recent_folders[name.lower()] = path

    def remember_file(self, name, path):
        self.recent_files[name.lower()] = path

    def get_folder(self, name):
        return self.recent_folders.get(name.lower())

    def get_file(self, name):
        return self.recent_files.get(name.lower())

# -------------------
# Long Term Memory
# -------------------
class LongTermMemory:
    def __init__(self, db_path="memory.sqlite"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self._migrate_add_embedding_column("folders")
        self._migrate_add_embedding_column("files")

        self.folder_names, self.faiss_folder_index = [], None
        self.file_names, self.faiss_file_index = [], None
        self.command_texts, self.faiss_command_index = [], None

    def close(self):
        self.conn.close()

    def _migrate_add_embedding_column(self, table_name):
        c = self.conn.cursor()
        c.execute(f"PRAGMA table_info({table_name})")
        columns = [row[1] for row in c.fetchall()]
        if "embedding" not in columns:
            c.execute(f"ALTER TABLE {table_name} ADD COLUMN embedding BLOB")
            self.conn.commit()
        c.close()

    def _create_tables(self):
        c = self.conn.cursor()
        c.execute(
            """CREATE TABLE IF NOT EXISTS folders
                     (name TEXT PRIMARY KEY, path TEXT, embedding BLOB)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS files
                     (name TEXT PRIMARY KEY, path TEXT, embedding BLOB)"""
        )
        c.execute(
            """CREATE TABLE IF NOT EXISTS commands
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, command TEXT, output TEXT, embedding BLOB, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)"""
        )
        self.conn.commit()

    def remember_folder(self, name, path, embedding=None):
        c = self.conn.cursor()
        emb_blob = (
            np.array(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        )
        c.execute(
            "REPLACE INTO folders (name, path, embedding) VALUES (?, ?, ?)",
            (name.lower(), path, emb_blob),
        )
        self.conn.commit()

    def remember_file(self, name, path, embedding=None):
        c = self.conn.cursor()
        emb_blob = (
            np.array(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        )
        c.execute(
            "REPLACE INTO files (name, path, embedding) VALUES (?, ?, ?)",
            (name.lower(), path, emb_blob),
        )
        self.conn.commit()

    def get_folder(self, name):
        c = self.conn.cursor()
        c.execute("SELECT path FROM folders WHERE name=?", (name.lower(),))
        row = c.fetchone()
        return row[0] if row else None

    def get_file(self, name):
        c = self.conn.cursor()
        c.execute("SELECT path FROM files WHERE name=?", (name.lower(),))
        row = c.fetchone()
        return row[0] if row else None

    def log_command(self, command, output, embedding=None):
        c = self.conn.cursor()
        emb_blob = (
            np.array(embedding, dtype=np.float32).tobytes() if embedding is not None else None
        )
        c.execute(
            "INSERT INTO commands (command, output, embedding) VALUES (?, ?, ?)",
            (command, output, emb_blob),
        )
        self.conn.commit()

    def _load_embeddings(self, table, text_col, emb_col):
        c = self.conn.cursor()
        try:
            c.execute(
                f"SELECT {text_col}, {emb_col} FROM {table} WHERE {emb_col} IS NOT NULL"
            )
        except sqlite3.OperationalError as e:
            if "no such column" in str(e):
                self._migrate_add_embedding_column(table)
                c.execute(
                    f"SELECT {text_col}, {emb_col} FROM {table} WHERE {emb_col} IS NOT NULL"
                )
            else:
                raise
        texts, embs = [], []
        for text, emb in c.fetchall():
            if emb:
                arr = np.frombuffer(emb, dtype=np.float32)
                texts.append(text)
                embs.append(arr)
        return texts, np.vstack(embs) if embs else np.zeros((0, 384), dtype=np.float32)

    def build_faiss_indices(self):
        # folders
        self.folder_names, folder_embs = self._load_embeddings("folders", "name", "embedding")
        if len(folder_embs) > 0:
            self.faiss_folder_index = faiss.IndexFlatL2(folder_embs.shape[1])
            self.faiss_folder_index.add(folder_embs)
        # files
        self.file_names, file_embs = self._load_embeddings("files", "name", "embedding")
        if len(file_embs) > 0:
            self.faiss_file_index = faiss.IndexFlatL2(file_embs.shape[1])
            self.faiss_file_index.add(file_embs)
        # commands
        self.command_texts, cmd_embs = self._load_embeddings("commands", "command", "embedding")
        if len(cmd_embs) > 0:
            self.faiss_command_index = faiss.IndexFlatL2(cmd_embs.shape[1])
            self.faiss_command_index.add(cmd_embs)

    def search_folders(self, query_vec, top_k=3):
        if self.faiss_folder_index is None:
            self.build_faiss_indices()
        if self.faiss_folder_index is None or self.faiss_folder_index.ntotal == 0:
            return []
        D, I = self.faiss_folder_index.search(np.array(query_vec, dtype=np.float32), top_k)
        return [(self.folder_names[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.folder_names)]

    def search_files(self, query_vec, top_k=3):
        if self.faiss_file_index is None:
            self.build_faiss_indices()
        if self.faiss_file_index is None or self.faiss_file_index.ntotal == 0:
            return []
        D, I = self.faiss_file_index.search(np.array(query_vec, dtype=np.float32), top_k)
        return [(self.file_names[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.file_names)]

    def search_commands(self, query_vec, top_k=3):
        if self.faiss_command_index is None:
            self.build_faiss_indices()
        if self.faiss_command_index is None or self.faiss_command_index.ntotal == 0:
            return []
        D, I = self.faiss_command_index.search(np.array(query_vec, dtype=np.float32), top_k)
        return [(self.command_texts[i], float(D[0][j])) for j, i in enumerate(I[0]) if i < len(self.command_texts)]

# -------------------
# Context Builder
# -------------------
def build_context(user_query, short_mem, long_mem, embedder):
    q_vec = embedder.encode([user_query])  # fastembed encoding
    related_folders = long_mem.search_folders(q_vec, top_k=3)
    related_files = long_mem.search_files(q_vec, top_k=3)
    related_cmds = long_mem.search_commands(q_vec, top_k=3)

    short = f"CWD: {short_mem.current_directory}\nLast command: {short_mem.last_command}\nLast output: {short_mem.last_output}"

    context = (
        f"{short}\n\n"
        f"Relevant folders: {related_folders}\n"
        f"Relevant files: {related_files}\n"
        f"Similar past commands: {related_cmds}\n"
    )
    return context
