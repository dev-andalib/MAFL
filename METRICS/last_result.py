import json
from pathlib import Path

# ✅ Change this to your folder
BIN_TEST_ROOT = Path(r"D:\MAFL\METRICS\LSTM with Attention\client_metrics\bin_test_metrics")
MUL_TEST_ROOT = Path(r"D:\MAFL\METRICS\LSTM with Attention\client_metrics\mul_test_metrics")

CALL_NO = 9
SEPARATOR = "-" * 26  # "--------------------------"

def extract_call9_metrics(json_data):
    # JSON is a list of records
    if isinstance(json_data, list):
        for item in json_data:
            if isinstance(item, dict) and item.get("call_number") == CALL_NO:
                return item.get("metrics")
        return None

    # JSON is a dict
    if isinstance(json_data, dict):
        # Dict itself might be a record
        if json_data.get("call_number") == CALL_NO:
            return json_data.get("metrics")

        # Search lists inside dict values
        for v in json_data.values():
            if isinstance(v, list):
                m = extract_call9_metrics(v)
                if m is not None:
                    return m

    return None

def main(root):
    json_files = sorted(root.rglob("*.json"))

    for fp in json_files:
        try:
            with fp.open("r", encoding="utf-8") as f:
                data = json.load(f)

            metrics = extract_call9_metrics(data)

            print(fp.name)

            if not metrics:
                print("CALL_9_MISSING")
            else:
                # Print all metric key-values in one straight line
                line = "  ".join(f"{k}={v}" for k, v in metrics.items())
                print(line)

            print(SEPARATOR)

        except Exception as e:
            print(fp.name)
            print(f"ERROR: {e}")
            print(SEPARATOR)

if __name__ == "__main__":
    print("\n" + "=" * 30 + "\n")
    print("BIN TEST RESULTS")
    print("\n" + "=" * 30 + "\n")
    main(BIN_TEST_ROOT)
    print("\n" + "=" * 30 + "\n")
    print("MUL TEST RESULTS")
    print("\n" + "=" * 30 + "\n")
    main(MUL_TEST_ROOT)

