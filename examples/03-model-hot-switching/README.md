# 03 - Model Hot-Switching

![header](./header.png)

> **The whole reason Anthropify exists.**
> One conversation. Four providers. The context never resets.

## What it shows

A single `[]anthropic.MessageParam` slice is passed through four turns.
Between turns, only `req.Model` changes:

| Turn | Provider | Model | Job |
|------|----------|-------|-----|
| 1 | Anthropic | `claude-sonnet-4-5` | plan a trip |
| 2 | Kimi | `kimi-k2-thinking` | turn the plan into Python |
| 3 | GLM | `glm-4.6` | translate to Japanese |
| 4 | Anthropic | `claude-sonnet-4-5` | summarise everything |

Because every provider speaks the Anthropic-canonical message shape, the
prior assistant turns are just appended to the slice and re-sent. There is
no per-provider context translation, no "history adapter", no shadow
state. The conversation is the conversation.

![flow](./flow.png)

## Run

```bash
set -a; source .env.e2e; set +a
go run ./examples/03-model-hot-switching
```

Required env: `ANTHROPIC_API_KEY`, `KIMI_API_KEY`, `GLM_API_KEY`.

## Sample Output

```
>>> Turn 1 -> Switching to Anthropic (planning) (claude-sonnet-4-5)
user: Plan a 3-day Tokyo trip in 5 short bullets. No prose.
assistant: # 3-Day Tokyo Trip
- Day 1: Senso-ji Temple → Tokyo Skytree → Shibuya Crossing
- Day 2: Tsukiji Market → Imperial Palace → Harajuku → Meiji Shrine
- Day 3: Teamlab → Odaiba → Shinjuku (Metro Government Building obs deck)
...

>>> Turn 2 -> Switching to Kimi (code) (kimi-k2-thinking)
user: Now write a tiny Python function that returns those 5 bullets...
assistant: ```python
def tokyo_bullets():
    return [
        "Day 1: Senso-ji Temple → Tokyo Skytree → Shibuya Crossing",
        ...

>>> Turn 3 -> Switching to GLM (translation) (glm-4.6)
user: 把上面这 5 条要点翻译成日语,保持要点编号。
assistant: 1. 1日目：浅草寺 → 東京スカイツリー → 渋谷スクランブル交差点
2. 2日目：築地場外市場での朝食 → 皇居東御苑 → 原宿 → 明治神宮
...

>>> Turn 4 -> Switching to Anthropic (summary) (claude-sonnet-4-5)
user: In 3 bullets, summarise the entire conversation so far in English.
assistant: - Created a 3-day Tokyo itinerary with 5 bullet points
- Converted the itinerary into a Python function returning a list
- Translated the 5 English bullet points into Japanese
```

Each turn sees the full history of every previous turn, regardless of
which provider produced it.

## Notes

- `kimi-k2-thinking` spends most of its token budget on internal
  reasoning. The example gives it 8k tokens for that turn so it has
  room left over to actually emit code.
- If a provider returns no text (e.g. budget exhausted), the example
  substitutes a sentinel string so the conversation can still continue.
  The Anthropic API rejects empty assistant turns.

## Source

[`main.go`](./main.go)
