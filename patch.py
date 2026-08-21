import re

with open('e:/Zomato_3/main.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update global dictionaries
content = content.replace(
    'restaurant_dish_insights = {}',
    'restaurant_dish_insights = {}\nrestaurant_aspects = {}\nrestaurant_trends = {}\nrestaurant_total_reviews = {}'
)

# 2. Add aspect keywords
content = content.replace(
    'CONS_KEYWORDS = ["bad", "worst", "terrible", "horrible", "late", "slow", "cold", "stale", "smell", "dirty", "rude", "expensive", "overpriced", "salty", "bland", "poor", "pathetic"]',
    '''CONS_KEYWORDS = ["bad", "worst", "terrible", "horrible", "late", "slow", "cold", "stale", "smell", "dirty", "rude", "expensive", "overpriced", "salty", "bland", "poor", "pathetic"]

ASPECT_MAP = {
    "Food": ["food", "taste", "delicious", "yummy", "chicken", "biryani", "panetr", "pizza", "burger", "meal", "dish", "stale", "bland"],
    "Service": ["service", "staff", "waiter", "manager", "slow", "fast", "rude", "polite", "late", "quick", "delivery", "wait"],
    "Ambience": ["ambience", "decor", "atmosphere", "music", "cozy", "place", "seating", "crowd", "dirty", "clean", "vibe"],
    "Price": ["price", "cost", "expensive", "cheap", "value", "money", "overpriced", "worth", "bill"]
}'''
)

# 3. Update the grouping loop to track aspects and trends
old_loop_init = '''        food_mentions = {k: {'pos': 0, 'total': 0} for k in FOOD_KEYWORDS}
        pros_counts = {k: 0 for k in PROS_KEYWORDS}
        cons_counts = {k: 0 for k in CONS_KEYWORDS}'''

new_loop_init = '''        food_mentions = {k: {'pos': 0, 'total': 0} for k in FOOD_KEYWORDS}
        pros_counts = {k: 0 for k in PROS_KEYWORDS}
        cons_counts = {k: 0 for k in CONS_KEYWORDS}
        
        aspect_counts = {k: {"pos": 0, "neu": 0, "neg": 0, "total": 0} for k in ASPECT_MAP.keys()}
        sentiment_trend = {}'''

content = content.replace(old_loop_init, new_loop_init)

# 4. Add logic inside the loop to track trends and aspects
old_loop_body = '''            global_sentiment_distribution[sentiment] += 1
            
            # Check food keywords
            for f in FOOD_KEYWORDS:'''

new_loop_body = '''            global_sentiment_distribution[sentiment] += 1
            
            # Extractyear from Time
            time_str = str(r_row.get('Time', ''))
            import re as regex
            year_match = regex.search(r'(20\\d\\d)', time_str)
            year = year_match.group(1) if year_match else "Unknown"
            
            if year != "Unknown":
                if year not in sentiment_trend:
                    sentiment_trend[year] = {"pos": 0, "neu": 0, "neg": 0}
                if sentiment == 'Positive': sentiment_trend[year]["pos"] += 1
                elif sentiment == 'Negative': sentiment_trend[year]["neg"] += 1
                else: sentiment_trend[year]["neu"] += 1
                
            # Aspect Analysis
            for aspect, kws in ASPECT_MAP.items():
                if any(kw in review_text for kw in kws):
                    aspect_counts[aspect]["total"] += 1
                    if sentiment == 'Positive': aspect_counts[aspect]["pos"] += 1
                    elif sentiment == 'Negative': aspect_counts[aspect]["neg"] += 1
                    else: aspect_counts[aspect]["neu"] += 1
            
            # Check food keywords
            for f in FOOD_KEYWORDS:'''

content = content.replace(old_loop_body, new_loop_body)

# 5. Save the tracked aspects and trends globally
old_save_global = '''        restaurant_pros[name] = top_pros
        restaurant_cons[name] = top_cons
        restaurant_dish_insights[name] = dish_insights[:5]'''

new_save_global = '''        restaurant_pros[name] = top_pros
        restaurant_cons[name] = top_cons
        restaurant_dish_insights[name] = dish_insights[:10]
        restaurant_aspects[name] = aspect_counts
        restaurant_trends[name] = sentiment_trend
        restaurant_total_reviews[name] = total_count'''

content = content.replace(old_save_global, new_save_global)

# 6. Update the /api/restaurants/{id} endpoint to return the new data
old_endpoint_return = '''            "dishInsights": restaurant_dish_insights.get(name, []),
            "sentimentDistribution": {
               "positive": pos_percent,
                "neutral": neu_percent,
                "negative": neg_percent
            },
            "reviews": restaurant_reviews,
            "menu": menu,
            "link": link
        }'''

new_endpoint_return = '''            "dishInsights": restaurant_dish_insights.get(name, []),
            "aspectAnalysis": restaurant_aspects.get(name, {}),
            "sentimentTrend": restaurant_trends.get(name, {}),
            "totalReviews": restaurant_total_reviews.get(name, len(restaurant_reviews)),
            "sentimentDistribution": {
               "positive": pos_percent,
                "neutral": neu_percent,
                "negative": neg_percent
            },
            "reviews": restaurant_reviews,
            "menu": menu,
            "link": link
        }'''

content = content.replace(old_endpoint_return, new_endpoint_return)

# 7. Update dish insights building logic to include mentions
old_dish_build = '''        for dish, stats in food_mentions.items():
            if stats['total'] >= 1: # Min mentions to be considered
                dish_score = int((stats['pos'] / stats['total']) * 100)
                dish_insights.append({"name": dish.capitalize(), "score": dish_score})'''

new_dish_build = '''        for dish, stats in food_mentions.items():
            if stats['total'] >= 1: # Min mentions to be considered
                dish_score = int((stats['pos'] / stats['total']) * 100)
                dish_insights.append({
                    "name": dish.capitalize(), 
                    "score": dish_score,
                    "mentions": stats['total']
                })'''

content = content.replace(old_dish_build, new_dish_build)

with open('e:/Zomato_3/main.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Backend patched successfully!")
