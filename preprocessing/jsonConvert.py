import json
import csv
import random

def json_to_csv(input_file: str, output_file: str, sample_size: int = None):
    """
    Convert newline-delimited JSON reviews to a CSV.
    
    Args:
        input_file (str): Path to input JSON file (one object per line).
        output_file (str): Path to save CSV output.
        sample_size (int, optional): Number of reviews to sample. 
                                     If None, use all reviews.
    """
    data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                # keep only reviews with non-empty text
                if obj.get("text") and obj.get("text").strip():
                    data.append(obj)

    # sample if needed
    if sample_size is not None and sample_size < len(data):
        data = random.sample(data, sample_size)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["business_name", "author_name", "review_text", "rating"])

        for entry in data:
            business_name = entry.get("gmap_id", "")
            author_name = entry.get("name", "")
            review_text = entry.get("text", "")
            rating = entry.get("rating", "")
            writer.writerow([business_name, author_name, review_text, rating])

    print(f"CSV saved to {output_file} with {len(data)} rows (only reviews with text)")

"""
def main():
    parser = argparse.ArgumentParser(description="Convert JSON reviews to CSV")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to the output CSV file")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of reviews to include (default: all)")
    args = parser.parse_args()

    json_to_csv(args.input_file, args.output_file, args.sample_size)
"""

if __name__ == "__main__":
    json_to_csv("data/input/review-Vermont_10.json", "data/input/review-Vermont_10.csv",10000)