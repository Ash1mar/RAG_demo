from typing import List

def simple_chunk(text: str, max_chars: int = 500, overlap: int = 50) -> List[str]:
    """
    朴素分段：按段落拆分，再对长段落按字符窗切块；足够支撑 MVP。
    """
    paras = [p.strip() for p in text.replace("\r", "").split("\n\n") if p.strip()]
    chunks: List[str] = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            start = 0
            while start < len(p):
                end = min(len(p), start + max_chars)
                chunks.append(p[start:end])
                if end == len(p):
                    break
                start = end - overlap
    return chunks
