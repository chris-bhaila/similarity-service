"""
Quick evaluation harness for the plagiarism-detection service.

Sends known pairs (exact copy / paraphrase / partial overlap / unrelated)
to a running instance of the service and checks the combined_score falls
in the expected band for that category.

Usage:
    venv/bin/python eval_similarity.py [base_url]
"""
import sys
import json
import urllib.request

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"

ORIGINAL = (
    "Machine learning models require large amounts of labeled data to "
    "achieve high accuracy on classification tasks. Overfitting occurs "
    "when a model learns noise in the training data instead of the "
    "underlying pattern."
)

CASES = [
    {
        "name": "exact_copy",
        "text": ORIGINAL,
        "expect_min": 0.90,
        "expect_max": 1.01,
    },
    {
        "name": "close_paraphrase",
        "text": (
            "ML models need lots of labeled training examples to reach "
            "good accuracy on classification problems. Overfitting "
            "happens when a model picks up noise in the training set "
            "rather than the real pattern."
        ),
        "expect_min": 0.55,
        "expect_max": 0.95,
    },
    {
        "name": "partial_overlap",
        "text": (
            "Overfitting is a common problem in statistics where a "
            "model fits the training data too closely. My favorite "
            "recipe for banana bread uses three ripe bananas and a "
            "cup of sugar."
        ),
        "expect_min": 0.20,
        "expect_max": 0.65,
    },
    {
        "name": "unrelated",
        "text": (
            "The city council voted on Tuesday to approve funding for "
            "a new public transit line connecting downtown to the "
            "airport by 2027."
        ),
        "expect_min": -1.0,
        "expect_max": 0.30,
    },
]


def check_submission(new_text, existing_texts):
    payload = {
        "new_submission": {"id": 0, "text": new_text},
        "existing_submissions": [
            {"id": i + 1, "text": t} for i, t in enumerate(existing_texts)
        ],
    }
    req = urllib.request.Request(
        f"{BASE_URL}/check-submission",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def main():
    texts = [c["text"] for c in CASES]
    result = check_submission(ORIGINAL, texts)

    # exact_copy will be compared against itself -> filter it out separately
    scores_by_id = {r["compared_submission_id"]: r for r in result["results"]}

    passed = 0
    print(f"{'case':<18} {'lexical':>8} {'semantic':>9} {'combined':>9} {'expected':>14}  result")
    for i, case in enumerate(CASES):
        r = scores_by_id[i + 1]
        combined = r["combined_score"]
        ok = case["expect_min"] <= combined <= case["expect_max"]
        passed += ok
        band = f"[{case['expect_min']:.2f},{case['expect_max']:.2f}]"
        print(
            f"{case['name']:<18} {r['lexical_score']:>8.4f} {r['semantic_score']:>9.4f} "
            f"{combined:>9.4f} {band:>14}  {'PASS' if ok else 'FAIL'}"
        )

    print(f"\n{passed}/{len(CASES)} cases within expected band")


if __name__ == "__main__":
    main()
