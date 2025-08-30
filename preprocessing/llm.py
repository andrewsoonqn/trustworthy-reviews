import pandas as pd
import ollama
import logging
import re
import json
from tqdm import tqdm
from count_flags import count_flags_df

# Generate the prompt for the LLM
def generate_prompt(review):
    try:
        prompt = f"""You are an AI tasked with evaluating the quality and relevance of Google maps location reviews. You are provided with the following review:
           {review} Your task is to evaluate if this review is inauthentic, an advertisement,
           irrelevant to the location, or an overly critical rant. In making your judgement,
           you may make reference to the other provided data columns where the review text has
           been pre-processed for the star rating given, its sentiment rating
           (-1 being the most negative sentiment and 1 being the most positive),
           subjectivity rating (0 being the most objective and 1 being the most subjective),
           language, review length, total number of exclamation marks in the review,
           total number of fully capitalised words in the review,
           and whether the review contains a url (which may be indicative of a promotion or advertisement).

            Inauthentic reviews may be reviews that are faked to artificially increase or decrease a business' ratings.
            Possible indicators may be reviews that give very high or low ratings,
            use strong but generic language with little concrete details
            (for example "Great place, amazing food, highly recommend!"), or are relatively short in length.

            Important: Do not mark a review as inauthentic unless there is strong evidence
            that it was written with deceptive intent. Authentic reviews may still be
            short, generic, overly positive/negative, or emotional. Only assign
            "Inauthentic: 1" if multiple strong indicators of deception are present,
            and avoid false positives.

            Irrelevant reviews include reviews that include content about things unrelated to the business,
            or reviews in languages other than English ("en").

            Advertisements may be reviews that use language common to advertisements
            or promotional material and may include urls.

            Rants are overly emotional or angry reviews that may not give a true representation of
            what to expect from the location. A high number of all-caps words and exclamation marks along with a strong negative sentiment may indicate a rant.

            The more of such indicative qualities a review has, the more likely it is to fall into a certain category.

           Provide your evaluation in the exact format below:
           Inauthentic: 1/0
           Irrelevant: 1/0
           Advertisement: 1/0
           Rant: 1/0
           Reason:

           If any one of the 1/0 fields are answered with a 1, please provide a reason under the "Reason:" field that concisely but adequately explains the violation. If there are multiple violations, please explain
           all of them clearly. Please do not include any of the markers in """
        logging.info(f"Prompt generated successfully.")
        return prompt
    except Exception as e:
        logging.error(f"Error generating prompt: {str(e)}")


# Organise the LLM's response to store the result in a library-like structure
def parse_result(response):
    """
    Parses the LLM response string and extracts evaluation fields.
    Expects the format:
        Inauthentic: 1/0
        Irrelevant: 1/0
        Advertisement: 1/0
        Rant: 1/0
        Violation: 1/0
        Reason: <text>
    """
    try:
        inauthentic = re.search(r"Inauthentic:\s*(1|0)", response)
        irrelevant = re.search(r"Irrelevant:\s*(1|0)", response)
        advertisement = re.search(r"Advertisement:\s*(1|0)", response)
        rant = re.search(r"Rant:\s*(1|0)", response)
        violation = re.search(r"Violation:\s*(1|0)", response)
        reason = re.search(r"Reason:\s*(.*)", response)

        evaluation = {
            "inauthentic": int(inauthentic.group(1)) if inauthentic else 0,
            "irrelevant": int(irrelevant.group(1)) if irrelevant else 0,
            "advertisement": int(advertisement.group(1)) if advertisement else 0,
            "rant": int(rant.group(1)) if rant else 0,
            "policy_violations": int(violation.group(1)) if violation else 0,
            "reason": reason.group(1).strip() if reason else ""
        }

        return evaluation

    except Exception as e:
        logging.error(f"Error parsing result: {str(e)}")
        return {
            "inauthentic": 0,
            "irrelevant": 0,
            "advertisement": 0,
            "rant": 0,
            "policy_violations": 0,
            "reason": ""
        }


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
                "inauthentic": 0,
                "irrelevant": 0,
                "advertisement": 0,
                "rant": 0,
                "policy_violations": 0,
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
                "inauthentic": eval_data.get("inauthentic", 0),
                "irrelevant": eval_data.get("irrelevant", 0),
                "advertisement": eval_data.get("advertisement", 0),
                "rant": eval_data.get("rant", 0),
                "policy_violations": eval_data.get("policy_violations", 0),
                "reason": eval_data.get("reason", "")
            }

            evaluation_results.append(review_dict)

        # Create a new DataFrame
        evaluated_df = pd.DataFrame(evaluation_results)

        # Save as CSV
        evaluated_df.to_csv(output_path, index=False, encoding="utf-8")

        flag_summary = count_flags_df(evaluated_df)
        logging.info(f"Flag summary: {flag_summary}")

        logging.info(f"Successfully wrote {len(evaluated_df)} reviews to {output_path}")
    except Exception as e:
        logging.error(f"Error compiling reviews: {str(e)}")

if __name__ == "__main__":
    compile_reviews("data/output/processed_reviews.csv", "data/llmOutput/llmevaluated_reviews_Kaggle_full.csv")
