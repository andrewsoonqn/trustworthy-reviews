import pandas as pd
import ollama
import logging
import re
import json


def generate_prompt(review):
    try:
        prompt = f"""You are an AI tasked with evaluating the quality and relevancy of Google location reviews. You are provided with the following review:
           {review.cleaned_text} Your task is to evaluate if this review is spam, an advertisement, irrelevant to the location, or a rant by a user who has likely never
           visited the location. In addition, you are required to analyse the sentiment of the review with the help of other information provided in {review}.

           Provide your evaluation in the exact format below:
           Sentiment: {review.sentiment}
           Spam or Advertisement: Yes/No
           Irrelevant: Yes/No
           Rant: Yes/No
           Violation: Yes/No
           Reason:
           
           If any one of the Yes/No fields are answered with a "Yes", the review should immediately be flagged as a violation. If a review is considered a violation,
           please provide a reason under the "Reason:" field that concisely but adequately explains the violation. If there are multiple violations, please explain
           all of them clearly. Do not give a reason if the review has no violations, just leave that field entirely blank.
            """
        logging.info(f"Prompt generated successfully.")
        return prompt
    except Exception as e:
        logging.error(f"Error generating prompt: {str(e)}")


def parse_result(response):
    try:
        sentiment = re.search(r"Sentiment:\s*(Positive|Negative|Neutral)", response)
        spam = re.search(r"Spam or Advertisement:\s*(Yes|No)", response)
        relevant = re.search(r"Irrelevant:\s*(Yes|No)", response)
        rant = re.search(r"Rant:\s*(Yes|No)", response)
        violation = re.search(r"Violation:\s*(Yes|No)", response)
        reason = re.search(r"Reason:\s*(.*)", response)

        evaluation = {
            "sentiment": {"value": sentiment.group(1) if sentiment else "Neutral"},
            "spam": {"value": spam.group(1) if spam else "No"},
            "relevant": {"value": relevant.group(1) if relevant else "Yes"},
            "rant": {"value": rant.group(1) if rant else "No"},
            "policy_violations": {"value": violation.group(1) if violation else "No"},
            "reason": reason.group(1) if reason else ""
        }

        return evaluation
    except Exception as e:
        logging.error(f"Error parsing result: {str(e)}")
        return {}


def evaluate_review(review):
    try:
        prompt = generate_prompt(review)
        result = ollama.generate(model="gemma3:4b", prompt=prompt)
        extracted = result.response
        evaluation = parse_result(extracted)
        logging.info("Evaluation completed successfully.")

        return {
            "review": review,
            "evaluation": evaluation
        }
    except Exception as e:
        logging.error(f"Error evaluating review: {str(e)}")
        return {
            "review": review,
            "evaluation": {
                "sentiment": {"value": "Unknown"},
                "spam": {"value": "Unknown"},
                "relevant": {"value": "Unknown"},
                "policy_violations": {"value": "Unknown"},
                "reason": ""
            }
        }

def compile_reviews(input_path, output_path):
    try:
        df = pd.read_csv(input_path)

        evaluation_results = []

        with open(output_path, "w") as json_file:
            for i, row in enumerate(df.itertuples()):
                text = row.cleaned_text
                evaluation = evaluate_review(row) or {}
                review_dict = {
                    "review": text,
                    "evaluation": evaluation
                }

                json_str = json.dumps(review_dict, ensure_ascii=False, indent=4)
                json_lines = json_str.splitlines()
                indented_lines = ["    " + line for line in json_lines]
                json_file.write("\n".join(indented_lines))

                if i < len(df) - 1:
                    json_file.write(",\n\n")
                else:
                    json_file.write("\n")

            json_file.write("]")

    except Exception as e:
        logging.error(f"Error compiling reviews: {str(e)}")

if __name__ == "__main__":
    compile_reviews("data/output/processed_reviews.csv", "data/output/llmevaluated_reviews.json")