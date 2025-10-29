#!/usr/bin/env bash
set -euo pipefail

BASE="http://127.0.0.1:8000"

echo "=== Ingest sample texts (no jq required) ==="
curl -sS -X POST "$BASE/ingest" -H "Content-Type: application/json" -d '{
  "doc_id":"doc1",
  "text":"RAG（Retrieval-Augmented Generation）是一种结合外部知识库的生成方法。通过检索相关文档，模型可以回答更准确的问题。"
}' >/dev/null

curl -sS -X POST "$BASE/ingest" -H "Content-Type: application/json" -d '{
  "doc_id":"doc2",
  "text":"BM25 是一种传统的基于关键词的检索算法。相比语义搜索，它依赖词频与逆文档频率（TF-IDF）计算相关性。"
}' >/dev/null

curl -sS -X POST "$BASE/ingest" -H "Content-Type: application/json" -d '{
  "doc_id":"doc3",
  "text":"混合检索（Hybrid Retrieval）结合了向量搜索与关键词搜索的优势，可在相关性和召回率之间取得平衡。"
}' >/dev/null

echo -e "\n=== Vector Search ==="
curl -sS --get "$BASE/search" --data-urlencode "q=检索 算法" --data-urlencode "k=5"

echo -e "\n\n=== Keyword Search ==="
curl -sS --get "$BASE/search_kw" --data-urlencode "q=检索 算法" --data-urlencode "k=5"

echo -e "\n\n=== Hybrid Search (alpha=0.5) ==="
curl -sS --get "$BASE/search_hybrid" --data-urlencode "q=检索 算法" --data-urlencode "k=5" --data-urlencode "alpha=0.5"

echo -e "\n\n=== Answer Generation ==="
curl -sS --get "$BASE/answer" --data-urlencode "q=RAG 的主要思想是什么" --data-urlencode "k=5" --data-urlencode "max_chars=300"
echo
