def build_report(context: dict) -> dict:
    return {"status": "generated", "context_keys": list(context.keys())}
