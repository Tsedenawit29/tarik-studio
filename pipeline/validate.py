"""Stage 0: validate storyboard.json against the schema and check consent gate + word budget."""
from common import ROOT, load_env, load_storyboard


def main() -> None:
    load_env()
    sb = load_storyboard(validate=True)
    words = sum(len(s["narration"]["text"].split()) for s in sb["shots"])
    print(f"OK: {len(sb['shots'])} shots, {words} narration words, language={sb['language']}")
    if sb["character"].get("real_person"):
        ethics = (ROOT / "ETHICS.md").read_text(encoding="utf-8") if (ROOT / "ETHICS.md").exists() else ""
        if "CONSENT: GRANTED" not in ethics:
            raise SystemExit("character.real_person is true but ETHICS.md has no 'CONSENT: GRANTED' record. "
                             "Record consent before generating anything of a real person.")


if __name__ == "__main__":
    main()
