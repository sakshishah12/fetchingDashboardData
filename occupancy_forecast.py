import google.generativeai as genai
from datetime import datetime, timedelta

# Configure Gemini
genai.configure(api_key="")  

# Initialize Model
model = genai.GenerativeModel('gemini-2.0-flash')  

def occupancy_forecast_agent(events_json, competitor_pricing_json, hotel_name, hotel_location, confirmed_bookings):
    """
    Predicts occupancy rates for the next 7 days based on events and competitor pricing.
    """
    start_date = datetime.now() + timedelta(days=1)
    end_date = start_date + timedelta(days=6)
    date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"

    prompt = f"""
You are a hotel revenue management AI specializing in occupancy forecasting.

Hotel: {hotel_name}
Location: {hotel_location}
Forecast Period: {date_range}

Based on the following data:
- Upcoming Events: {events_json}
- Competitor Pricing: {competitor_pricing_json}
- Confirmed bookings percentage for the upcoming week:{confirmed_bookings}

Please provide a JSON list forecasting the occupancy rate for each day in the forecast period. For each day, include:
- "date": Date in YYYY-MM-DD format.
- "forecasted_occupancy_percentage": Predicted occupancy rate as a percentage.
- "reasoning": Brief explanation for the forecast.
- "actionable_insights": Actionable tips based on the forecast

Ensure the output is a JSON array with one entry per day.
"""
    response = model.generate_content(
        prompt,
        generation_config={
            "temperature": 0.5,
            "response_mime_type": "application/json"
        }
    )

    return response.text


def occupancy_forecast_reviewer_agent(occupancy_json, events_json, competitor_pricing_json):
    """
    Reviews the occupancy forecast JSON output for correctness, logic, and structure.
    """
    reviewer_prompt = f"""
You are an expert hotel revenue forecasting reviewer.

Your job is to critically evaluate the following occupancy forecast JSON, based on:
- Local events: {events_json}
- Competitor pricing: {competitor_pricing_json}

### Instructions:
- Ensure each forecast entry includes:
  - "date" (YYYY-MM-DD)
  - "forecasted_occupancy_percentage" (0–100 only)
  - "reasoning" (must mention event impact, booking trends, or competitor pricing if relevant)
  - "actionable_insights" (should be practical and hotel-specific)

- Identify issues such as:
  - Illogical occupancy numbers (e.g. 95% on a day with no events and high competition)
  - Missing or vague reasoning
  - Unrealistic fluctuations
  - Missing fields

If you find problems, fix them.

### Output Format:
Respond in JSON with two keys:
{{
  "review": "Brief feedback on accuracy and structure",
  "corrected_occupancy_forecast": [ ... corrected entries if applicable ... ]
}}

Here is the forecast to review:

{occupancy_json}
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
hotel_location = "Stony Brook, NY"
events_json=[{"event_name": "Wine Tasting Event", "date": "2025-05-01", "location": "Hilton Garden Inn, Stony Brook", "short_description": "Enjoy a selection of fine wines paired with appetizers. Open to hotel guests and the public."}, {"event_name": "Live Music - Acoustic Night", "date": "2025-05-02", "location": "The Bench Bar & Grill (adjacent to Hilton Garden Inn)", "short_description": "Relax and unwind with live acoustic music at The Bench."}, {"event_name": "Stony Brook Farmers Market", "date": "2025-05-03", "location": "Stony Brook Village Center", "short_description": "Browse fresh produce, baked goods, and local crafts. Short walk from the hotel."}, {"event_name": "Long Island Ducks Baseball Game", "date": "2025-05-04", "location": "Fairfield Properties Ballpark, Central Islip", "short_description": "Catch a Long Island Ducks baseball game. Approximately 30-minute drive from the hotel."}, {"event_name": "Guided Kayak Tour", "date": "2025-05-05", "location": "West Meadow Beach, Stony Brook", "short_description": "Explore the beautiful waters of Long Island Sound on a guided kayak tour. Short drive from the hotel."}, {"event_name": "Art Exhibit Opening", "date": "2025-05-06", "location": "Stony Brook University Art Gallery", "short_description": "Attend the opening reception for a new art exhibit. Short walk from the hotel."}, {"event_name": "Sunday Brunch Buffet", "date": "2025-05-07", "location": "Hilton Garden Inn, Stony Brook", "short_description": "Enjoy a delicious Sunday brunch buffet at the hotel."}]
competitor_pricing_json=[{"hotel_name": "Holiday Inn Express Stony Brook", "price_per_night_usd": "189", "star_rating": "3", "distance_miles": "1.2", "short_description": "Modern mid-scale hotel near university."}, {"hotel_name": "Three Village Inn", "price_per_night_usd": "225", "star_rating": "3.5", "distance_miles": "2.5", "short_description": "Charming historic inn with suites."}, {"hotel_name": "Danfords Hotel and Marina", "price_per_night_usd": "299", "star_rating": "4", "distance_miles": "5.7", "short_description": "Waterfront hotel with marina views."}, {"hotel_name": "The Harborfront Inn", "price_per_night_usd": "165", "star_rating": "2.5", "distance_miles": "4.9", "short_description": "Budget-friendly option near the harbor."}, {"hotel_name": "Residence Inn Long Island Port Jefferson", "price_per_night_usd": "210", "star_rating": "3", "distance_miles": "7.1", "short_description": "Extended-stay hotel with kitchenettes."}, {"hotel_name": "Soundview Greenport", "price_per_night_usd": "349", "star_rating": "4.5", "distance_miles": "9.8", "short_description": "Luxury boutique hotel with harbor views."}]
confirmed_bookings=[
  {"date": "2025-05-01", "confirmed_booking_percent": 60},
  {"date": "2025-05-02", "confirmed_booking_percent": 65},
  {"date": "2025-05-03", "confirmed_booking_percent": 58},
  {"date": "2025-05-04", "confirmed_booking_percent": 70},
  {"date": "2025-05-05", "confirmed_booking_percent": 62},
  {"date": "2025-05-06", "confirmed_booking_percent": 75},
  {"date": "2025-05-07", "confirmed_booking_percent": 68}
]


occupancy_json = occupancy_forecast_agent(events_json, competitor_pricing_json, hotel_name, hotel_location,confirmed_bookings)
forecast_output =  occupancy_forecast_reviewer_agent(events_json, competitor_pricing_json,occupancy_json)
print(forecast_output)
