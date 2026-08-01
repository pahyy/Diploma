### Full run (~211 LLM prompts + unit tests)
```
pytest
```

### Representatives only (~28 LLM calls)
```
pytest -m representative
```

### Single batch
```
pytest -m batch04
```