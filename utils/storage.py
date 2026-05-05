import pandas as pd
import sqlite3
from pathlib import Path

def save_to_csv(df: pd.DataFrame, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[Storage] CSV sauvegardé : {path} ({len(df)} lignes)")

def save_to_sqlite(df: pd.DataFrame, path: str, table: str = "products"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()
    print(f"[Storage] SQLite sauvegardé : {path} (table '{table}')")

def load_from_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")