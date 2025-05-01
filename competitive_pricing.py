import google.generativeai as genai

# Step 1: Configure Gemini
genai.configure(api_key="")

# Step 2: Initialize Model
model = genai.GenerativeModel('gemini-1.5-pro')  # or 'gemini-2.5-pro'

# Step 3: Define Competitor Pricing Agent

def competitor_pricing_agent(hotel_name, hotel_location, date_range):
    """Fetches competitor hotel pricing around a given hotel."""
    
    prompt = f"""
You are a hotel pricing expert who has access to real-time hotel rates through various websites like Booking.com, Expedia, and Hotels.com.

I am the owner of {hotel_name}, located at {hotel_location}.
Today is {date_range.split(' to ')[0]}.

Your task:
- Find the pricing of my competitor hotels nearby for the date range {date_range}.
- Include only legitimate nearby hotels (within 10 miles).
- Use realistic names, prices, and ratings.
- If needed, make reasonable assumptions based on typical hotel market rates.

### Format:
Strictly reply in JSON array format where each object contains:
- "hotel_name"
- "price_per_night_usd"
- "star_rating"
- "distance_miles" (from my hotel)
- "short_description" (few words about hotel type: luxury, budget, boutique etc.)

### Example:

[
  {{
    "hotel_name": "Holiday Inn Express Stony Brook",
    "price_per_night_usd": 189,
    "star_rating": 3,
    "distance_miles": 1.2,
    "short_description": "Modern mid-scale hotel near university."
  }},
  ...
]

Only return the JSON, no explanations.
"""
    
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.3,
            "response_mime_type": "application/json"
        }
    )
    
    return response.text


# Step 4: Define the Reviewer Agent

def competitor_pricing_reviewer_agent(pricing_json):
    """Reviews the competitor hotel pricing JSON output."""
    
    reviewer_prompt = f"""
You are an expert hotel pricing reviewer.

Your task is to carefully review the following JSON list of competitor hotel pricing for a hotel owner.

### Instructions:
- Ensure that each entry contains:
  - "hotel_name"
  - "price_per_night_usd"
  - "star_rating"
  - "distance_miles"
  - "short_description"
- Check that:
  - Prices make sense based on hotel type (luxury hotels should not be cheaper than budget ones).
  - Star ratings are logically aligned with pricing.
  - Distance values are reasonable.
  - Descriptions match the price and rating (e.g., a $500 hotel should not be described as "budget").
- Spot unrealistic pricing if any (example: 5-star hotel priced at $80/night).
- If you find any issues, suggest a corrected JSON.

### Output Format:
Strict JSON structure:
{{
  "review": "Brief feedback text",
  "corrected_competitor_pricing": [ ... corrected list if needed else original ... ]
}}

Here is the competitor pricing data:

{pricing_json}
"""
    
    response = model.generate_content(
        reviewer_prompt,
        generation_config={
            "temperature": 0.2,
            "response_mime_type": "application/json"
        }
    )
    
    return response.text


hotel_name = "Hilton Garden Inn"
hotel_location = "Stony Brook, New York"
date_range = "2025-05-01 to 2025-05-07"

# 1. Get Competitor Pricing
competitor_pricing = competitor_pricing_agent(hotel_name, hotel_location, date_range)

# 2. Review Competitor Pricing
reviewed_competitor_pricing = competitor_pricing_reviewer_agent(competitor_pricing)

# 3. Print outputs
print("=== Competitor Pricing Output ===")
print(competitor_pricing)

print("\n=== Reviewed Competitor Pricing Output ===")
print(reviewed_competitor_pricing)
