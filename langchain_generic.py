from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import json

# ==== Configure Gemini via LangChain ====
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=""
)

# === Expert Prompt Template ===
expert_prompt_template = PromptTemplate.from_template("""
You are an expert AI tasked with the following:

{task_description}

{reviewer_feedback}

### Output Format:
{output_format}
""")

# === Reviewer Prompt Template ===
reviewer_prompt_template = PromptTemplate.from_template("""
You are a reviewer.

Review the following generated output according to these instructions:

{review_instructions}

### Data:
{data_to_review}
""")

# === Expert Agent Chain ===
def create_expert_chain(llm):
    return expert_prompt_template | llm | (lambda x: x.content)

# === Reviewer Agent Chain ===
def create_reviewer_chain(llm):
    def parse_json(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "approved": False,
                "feedback": "Malformed reviewer JSON.",
                "suggested_changes": "Ensure correct formatting and required keys."
            }

    return reviewer_prompt_template | llm | (lambda x: parse_json(x.content))

# === Feedback Loop Runner ===
def run_feedback_loop(task_description, output_format, review_instructions, max_iters=5):
    expert_chain = create_expert_chain(llm)
    reviewer_chain = create_reviewer_chain(llm)

    feedback = ""
    for i in range(max_iters):
        print(f"\n--- Iteration {i + 1} ---")
        expert_input = {
            "task_description": task_description,
            "reviewer_feedback": f"### Reviewer Feedback:\n{feedback}" if feedback else "",
            "output_format": output_format
        }
        result = expert_chain.invoke(expert_input)

        review_input = {
            "review_instructions": review_instructions,
            "data_to_review": result
        }
        review = reviewer_chain.invoke(review_input)

        print(f"Reviewer Feedback: {review.get('feedback')}")
        if review.get("approved"):
            print("Approved.")
            return result

        feedback = review.get("suggested_changes") or review.get("feedback")

    print("Max iterations reached.")
    return result

# === Task Setup ===
task_description = """
You are an event aggregator AI. Your task is to find realistic and important events happening near Times Square, New York City for the next 7 days starting from tomorrow: 2025-05-10.

Requirements:
- Each event must include:
  - "event_name"
  - "date"
  - "location"
  - "short_description"
- Return only a valid JSON array.
- Sort events by ascending date.
"""

output_format = """
[
  {
    "event_name": "...",
    "date": "...",
    "location": "...",
    "short_description": "..."
  },
  ...
]
"""

review_instructions = """
Review the JSON list of events:
- Ensure correct JSON formatting.
- Each event must include: "event_name", "date", "location", "short_description".
- Dates must be in the next 7 days from 2025-05-16 and sorted in ascending order.
- Events must be realistic and appropriate for New York City.
"""

# === Run the LangChain System ===
final_json = run_feedback_loop(
    task_description=task_description,
    output_format=output_format,
    review_instructions=review_instructions,
    max_iters=3
)

# === Final Output ===
print("\nFinal Approved JSON:\n")
print(final_json)
