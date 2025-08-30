import pandas as pd
import ollama
import logging
import re
import json
from tqdm import tqdm

# Generate the prompt for the LLM
def generate_prompt(review):
    try:
        prompt = f"""You are an AI tasked with evaluating the quality and relevancy of Google location reviews. You are provided with the following review:
           {review.cleaned_text} Your task is to evaluate if this review is spam, an advertisement, irrelevant to the location, or a rant by a user who has likely never
           visited the location. In addition, you are required to analyse the sentiment of the review with the help of other information provided in {review}.

           Provide your evaluation in the exact format below:
           Sentiment: {review.sentiment}
           Advertisement: 1/0
           Irrelevant: 1/0
           Rant: 1/0
           Violation: 1/0
           Reason:
           
           If any one of the Yes/No fields are answered with a "Yes", the review should immediately be flagged as a violation. If a review is considered a violation,
           please provide a reason under the "Reason:" field that concisely but adequately explains the violation. If there are multiple violations, please explain
           all of them clearly. Do not give a reason if the review has no violations, just leave that field entirely blank.
            """
        logging.info(f"Prompt generated successfully.")
        return prompt
    except Exception as e:
        logging.error(f"Error generating prompt: {str(e)}")


# Organise the LLM's response to store the result in a library-like structure
def parse_result(response):
    try:
        sentiment = re.search(r"Sentiment:\s*(Positive|Negative|Neutral)", response)
        spam = re.search(r"Advertisement:\s*(1|0)", response)
        irrelevant = re.search(r"Irrelevant:\s*(1|0)", response)
        rant = re.search(r"Rant:\s*(1|0)", response)
        violation = re.search(r"Violation:\s*(1|0)", response)
        reason = re.search(r"Reason:\s*(.*)", response)

        evaluation = {
            "sentiment": sentiment.group(1) if sentiment else "Neutral",
            "advertisement": int(spam.group(1)) if spam else 0,
            "irrelevant": int(irrelevant.group(1)) if irrelevant else 0,
            "rant": int(rant.group(1)) if rant else 0,
            "policy_violations": int(violation.group(1)) if violation else 0,
            "reason": reason.group(1).strip() if reason else ""
        }

        return evaluation
    except Exception as e:
        logging.error(f"Error parsing result: {str(e)}")
        return {}


# Main function applied to each review that generates the prompt and structures the response properly
def evaluate_review(review):
    try:
        prompt = generate_prompt(review)
        result = ollama.generate(model="gemma3:4b", prompt=prompt)
        extracted = result["response"]
        evaluation = parse_result(extracted)
        logging.info("Evaluation completed successfully.")

        return {
            "review": review.cleaned_text,
            "evaluation": evaluation
        }
    except Exception as e:
        logging.error(f"Error evaluating review: {str(e)}")
        return {
            "review": review.cleaned_text,
            "evaluation": {
                "sentiment": "Unknown",
                "advertisement": -1,
                "irrelevant": -1,
                "rant": -1,
                "policy_violations": -1,
                "reason": ""
            }
        }

# Collate the individual responses from the LLM into a single .json file with organised data
def compile_reviews(input_path, output_path, sample_size=None):
    try:
        df = pd.read_csv(input_path)

        if sample_size:
            df = df.sample(n=sample_size, random_state=42)

        evaluation_results = []

        for _, row in tqdm(list(enumerate(df.itertuples())), total=len(df), desc="Evaluating reviews"):
            evaluation = evaluate_review(row) or {}
            eval_data = evaluation["evaluation"]

            review_dict = {
                "business_name": getattr(row, "business_name", ""),
                "author_name": getattr(row, "author_name", ""),
                "review_text": getattr(row, "cleaned_text", ""),
                "rating": getattr(row, "rating", None),
                "sentimentNum": getattr(row, "sentiment", "Neutral"),
                "subjectivity": getattr(row, "subjectivity", ""),
                "lang": getattr(row, "lang", ""),
                "review_length": getattr(row, "review_length", ""),
                "exclaim_count": getattr(row, "exclaim_count", 0),
                "caps_count": getattr(row, "caps_count", 0),
                "contains_url": getattr(row, "contains_url", 0),
                # extra evaluation fields
                "sentiment": eval_data.get("sentiment", "Neutral"),
                "spam": eval_data.get("spam", 0),
                "irrelevant": eval_data.get("irrelevant", 0),
                "rant": eval_data.get("rant", 0),
                "policy_violations": eval_data.get("policy_violations", 0),
            }

            evaluation_results.append(review_dict)

        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(evaluation_results, json_file, ensure_ascii=False, indent=4)

        logging.info(f"Successfully wrote {len(evaluation_results)} reviews to {output_path}")
    except Exception as e:
        logging.error(f"Error compiling reviews: {str(e)}")

if __name__ == "__main__":
    compile_reviews("data/output/processed_reviews_400.csv", "data/llmOutput/llmevaluated_reviews_Kaggle_400.json")