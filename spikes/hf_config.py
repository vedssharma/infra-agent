"""M0 spike: read model shape from the Hub config.json (no weight download) and estimate param count."""
import sys

import httpx


def main(repo: str) -> None:
    cfg = httpx.get(f"https://huggingface.co/{repo}/resolve/main/config.json", follow_redirects=True, timeout=30)
    cfg.raise_for_status()
    c = cfg.json()
    h, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    inter = c.get("intermediate_size", 4 * h)
    heads, kv = c["num_attention_heads"], c.get("num_key_value_heads", c["num_attention_heads"])
    head_dim = h // heads
    attn = h * h + 2 * h * kv * head_dim + h * h
    mlp = 3 * h * inter
    emb = V * h * (1 if c.get("tie_word_embeddings") else 2)
    params = L * (attn + mlp) + emb
    print(f"{repo}: hidden={h} layers={L} vocab={V} inter={inter} heads={heads} kv_heads={kv}")
    print(f"approx params: {params/1e9:.2f}B  | bf16 weights: {params*2/2**30:.2f} GiB")
    print(f"logits @ batch*seq=4096 tokens (fp32): {4096*V*4/2**30:.2f} GiB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-1.5B")
