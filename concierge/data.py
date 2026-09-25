"""Synthetic knowledge base branded as Four Seasons Hotels and Resorts.

DEMO DATA ONLY: all venues, hours, prices, policies and guests are invented and do not
reflect the real Four Seasons properties. Not affiliated with or endorsed by Four Seasons.

Restaurant and spa records feed BOTH the SQL tables (structured tools) and the
KB documents (RAG), so the two never disagree.
"""
import random
from collections import defaultdict
from datetime import date, timedelta

BRAND = "Four Seasons Hotels and Resorts"

HOTELS = {
    "MTL": dict(name="Four Seasons Hotel Montréal", city="Montréal, Québec, Canada", tz="America/Toronto",
                lat=45.5075, lon=-73.5540, currency="CAD", check_in="15:00", check_out="12:00"),
    "WHI": dict(name="Four Seasons Resort Whistler", city="Whistler, British Columbia, Canada", tz="America/Vancouver",
                lat=50.1163, lon=-122.9574, currency="CAD", check_in="16:00", check_out="11:00"),
    "TOR": dict(name="Four Seasons Hotel Toronto", city="Toronto, Ontario, Canada", tz="America/Toronto",
                lat=43.6719, lon=-79.3903, currency="CAD", check_in="15:00", check_out="12:00"),
}

_R = ["hotel_id", "venue", "cuisine", "meal_period", "days", "open_time", "last_seating",
      "reservation_required", "dress_code", "kid_friendly", "location"]
RESTAURANTS = [dict(zip(_R, r)) for r in [
    ("MTL", "Verrière", "French brasserie", "breakfast", "Daily", "06:30", "10:30", False, "Casual", True, "Lobby level glass atrium"),
    ("MTL", "Verrière", "French brasserie", "weekend brunch", "Sat-Sun", "10:30", "14:00", False, "Casual", True, "Lobby level glass atrium"),
    ("MTL", "Verrière", "French brasserie", "dinner", "Daily", "17:30", "21:30", False, "Smart casual", True, "Lobby level glass atrium"),
    ("MTL", "Le Quai Bar", "Cocktails and small plates", "bar", "Daily", "16:00", "23:30", False, "Smart casual", False, "Ground floor, St. Lawrence view"),
    ("MTL", "Sakura Room", "Japanese omakase (12 seats)", "dinner", "Tue-Sat", "18:00", "20:30", True, "Smart casual", False, "Level 2"),
    ("MTL", "In-Room Dining", "All-day menu", "in-room", "Daily", "00:00", "23:59", False, "-", True, "Your room"),
    ("WHI", "Summit Grill", "Canadian steakhouse", "breakfast", "Daily", "07:00", "11:00", False, "Casual", True, "Main lodge"),
    ("WHI", "Summit Grill", "Canadian steakhouse", "dinner", "Daily", "17:00", "21:30", True, "Mountain casual", True, "Main lodge"),
    ("WHI", "Alpenglow Lounge", "Swiss fondue and après-ski", "après-ski and dinner", "Daily", "15:00", "22:30", False, "Casual", True, "Slope-side terrace and fireplace room"),
    ("WHI", "Base Camp Café", "Coffee, pastries, grab-and-go", "all-day café", "Daily", "06:00", "15:00", False, "Casual", True, "Ski valet level"),
    ("WHI", "In-Room Dining", "All-day menu", "in-room", "Daily", "06:00", "23:00", False, "-", True, "Your room"),
    ("TOR", "Aria", "Modern Italian", "breakfast", "Daily", "06:45", "10:30", False, "Casual", True, "Lobby level"),
    ("TOR", "Aria", "Modern Italian", "lunch", "Mon-Fri", "11:30", "14:30", False, "Smart casual", True, "Lobby level"),
    ("TOR", "Aria", "Modern Italian", "dinner", "Daily", "17:30", "22:00", False, "Smart casual", True, "Lobby level"),
    ("TOR", "Yorkville Lounge", "Cocktails and light bites", "lounge", "Daily", "12:00", "23:00", False, "Smart casual", False, "Ground floor, Yorkville Avenue side"),
    ("TOR", "Kaiseki Hana", "Japanese kaiseki tasting menu (10 seats)", "dinner", "Wed-Sun", "18:00", "20:00", True, "Smart casual", False, "Level 2"),
    ("TOR", "In-Room Dining", "All-day menu", "in-room", "Daily", "00:00", "23:59", False, "-", True, "Your room"),
]]

SPA = {
    "MTL": ("Spa Lumière", "09:00", "21:00", [("Signature Lumière Massage", 60, 210), ("Nordic Thermal Circuit", 90, 85), ("Maple Glow Facial", 75, 190)]),
    "WHI": ("Glacier Spa", "10:00", "21:00", [("Après-Ski Recovery Massage", 60, 220), ("Hot Stone Massage", 90, 295), ("Outdoor Hot Pools Pass", 120, 70)]),
    "TOR": ("Spa Yorkville", "09:00", "21:00", [("Deep Tissue Massage", 60, 230), ("Hydrotherapy Circuit", 90, 95), ("Signature Radiance Facial", 75, 210)]),
}

# (category, title, body) — hotel-specific prose. hotel_id "ALL" = brand-wide.
_DOCS = {
    "MTL": [
        ("overview", "Hotel overview", "A 120-room heritage hotel in a restored 1880s warehouse on Rue de la Commune in Old Montréal, facing the Old Port. Check-in 3:00 PM, check-out 12:00 PM. Concierge desk open 24 hours."),
        ("amenities", "Amenities", "Indoor saltwater pool (6:00 AM-10:00 PM), 24-hour fitness centre with Peloton bikes, rooftop terrace (May-October, 11:00 AM-10:00 PM), business centre and two meeting rooms, complimentary high-speed Wi-Fi, valet parking (CAD 48/night), bicycle loans in summer, Tesla and universal EV chargers."),
        ("family", "Families and children", "There is no dedicated kids club at the Montréal property. Families receive a welcome kit, cribs and rollaway beds on request (free), a children's menu at Verrière, and certified babysitting (CAD 30/hour, book 24 hours ahead). Family suites have connecting rooms. The concierge recommends the Montréal Science Centre and the Biodôme for children."),
        ("rainy_day", "Rainy-day ideas", "On rainy days: the Nordic Thermal Circuit at Spa Lumière, the indoor saltwater pool, the Saturday 3:00 PM mixology class at Le Quai Bar (CAD 65), Pointe-à-Callière archaeology museum (5-minute walk), the Montréal Museum of Fine Arts (10 minutes by taxi), and the RÉSO underground city for shopping. Complimentary umbrellas at the door."),
        ("activities", "Local activities", "Walking tours of Old Montréal depart the lobby daily at 10:00 AM (CAD 35). Nearby: Notre-Dame Basilica (7-minute walk), Old Port promenade and zip line (summer), Jean-Talon Market, Mount Royal lookout. In winter, the Old Port skating rink is 3 minutes away."),
        ("transport", "Transportation", "Montréal-Trudeau Airport (YUL) is 25-35 minutes by car. Private airport transfer CAD 95 each way in a Mercedes sedan; book 24 hours ahead. Place-d'Armes metro station is a 6-minute walk."),
        ("policies", "Hotel policies (Montréal)", "Dogs and cats up to 25 lb are welcome, maximum one pet per room, fee CAD 75 per stay. Late check-out until 2:00 PM is CAD 60, subject to availability (free for Four Seasons Preferred Guest Gold and above)."),
        ("faq", "FAQ", "Breakfast is served at Verrière from 6:30 AM to 10:30 AM daily; weekend brunch runs 10:30 AM-2:00 PM. Luggage storage is free before check-in and after check-out. Laundry and pressing returned same day if dropped by 9:00 AM."),
    ],
    "WHI": [
        ("overview", "Hotel overview", "A 90-room ski-in/ski-out lodge at the base of Blackcomb Mountain in Whistler's Upper Village. Check-in 4:00 PM, check-out 11:00 AM. Ski valet opens 7:00 AM."),
        ("amenities", "Amenities", "Heated outdoor pool and three hot tubs (7:00 AM-10:00 PM), fitness centre with yoga studio, ski valet with boot warmers, equipment rental shop, games room, complimentary Wi-Fi, underground parking (CAD 40/night), guest shuttle around Whistler Village every 20 minutes (7:00 AM-11:00 PM)."),
        ("family", "Families and children", "Little Rangers Kids Club for ages 4-12 runs daily 9:00 AM-4:00 PM (CAD 85/day, lunch included) with nature crafts, snowshoeing in winter and mountain biking basics in summer. Teen Adventure Nights on Fridays 6:00-9:00 PM for ages 13-17. Babysitting CAD 32/hour. Family movie night in the games room Saturdays at 7:00 PM."),
        ("rainy_day", "Rainy-day ideas", "When it rains: Glacier Spa (the outdoor hot pools stay open in light rain), board games and billiards in the games room, fondue at Alpenglow Lounge, the Audain Art Museum (5-minute shuttle), the Squamish Lil'wat Cultural Centre, and the Meadow Park Sports Centre with indoor pool and climbing wall. The concierge can book a guided indoor climbing session (CAD 70)."),
        ("activities", "Local activities", "Winter: skiing and snowboarding on Whistler Blackcomb, snowshoe tours, dog sledding in the Callaghan Valley. Summer: Peak 2 Peak Gondola, lift-access mountain biking, Lost Lake trails, ziplining with Ziptrek, and canoeing the River of Golden Dreams."),
        ("transport", "Transportation", "Vancouver International Airport (YVR) is about 2.5 hours by car. The Four Seasons private transfer from YVR is CAD 420 each way for up to 5 guests (book 48 hours ahead). Shared shuttle options run from the airport several times daily. Guest village shuttle is complimentary."),
        ("policies", "Hotel policies (Whistler)", "Pets are not permitted at the Whistler Lodge, except service animals. Ski lockers are complimentary. Winter cancellation (December-March) requires 14 days' notice."),
        ("faq", "FAQ", "Breakfast at Summit Grill starts at 7:00 AM daily until 11:00 AM; Base Camp Café opens at 6:00 AM for coffee and grab-and-go. Lift tickets can be purchased at the concierge desk."),
    ],
    "TOR": [
        ("overview", "Hotel overview", "A 200-room contemporary hotel in Yorkville, steps from Bloor Street shopping. Check-in 3:00 PM, check-out 12:00 PM. Concierge desk open 24 hours."),
        ("amenities", "Amenities", "Indoor pool with outdoor sun terrace (6:00 AM-10:00 PM), 24-hour fitness centre with yoga studio, business centre and ballroom, complimentary high-speed Wi-Fi, valet parking (CAD 65/night), complimentary house car within 3 km (7:00 AM-10:00 PM), EV chargers."),
        ("family", "Families and children", "Little Explorers weekend kids program for ages 5-12 runs Saturday and Sunday 10:00 AM-3:00 PM (CAD 70/day, lunch included) with crafts and a museum scavenger hunt. Cribs and rollaway beds free on request, children's menu at Aria, babysitting CAD 35/hour (book 24 hours ahead)."),
        ("rainy_day", "Rainy-day ideas", "On rainy days: the Hydrotherapy Circuit at Spa Yorkville, the indoor pool, the Royal Ontario Museum (5-minute walk), the Art Gallery of Ontario (10 minutes by house car), the Hockey Hall of Fame, and the PATH underground walkway for shopping downtown. Aria runs a pasta-making class Saturdays at 2:00 PM (CAD 90). Complimentary umbrellas at the door."),
        ("activities", "Local activities", "Yorkville galleries and boutiques on the doorstep. Nearby: CN Tower (15 minutes by car), St. Lawrence Market, the Distillery District, the Toronto Islands ferry (summer), Casa Loma, and Blue Jays or Maple Leafs games (the concierge can source tickets)."),
        ("transport", "Transportation", "Toronto Pearson Airport (YYZ) is 30-45 minutes by car; private transfer CAD 150 each way, book 24 hours ahead. The UP Express train links Pearson to Union Station in 25 minutes. Billy Bishop Airport (YTZ) is 20 minutes by car. Bay subway station is a 3-minute walk."),
        ("policies", "Hotel policies (Toronto)", "Dogs up to 30 lb are welcome, maximum one pet per room, fee CAD 100 per stay. Kaiseki Hana is for guests 12 and older. Late check-out until 2:00 PM is CAD 75, subject to availability."),
        ("faq", "FAQ", "Breakfast at Aria starts at 6:45 AM daily and runs until 10:30 AM. Luggage storage is free before check-in and after check-out. Same-day pressing if dropped by 9:00 AM."),
    ],
    "ALL": [
        ("policies", "Brand cancellation and payment policy", "Reservations can be cancelled free of charge up to 72 hours before arrival unless a property states otherwise; later cancellations are charged one night. A credit card is required at check-in for incidentals. All Four Seasons properties are non-smoking."),
        ("loyalty", "Four Seasons Preferred Guest loyalty program", "Four Seasons Preferred Guest has three tiers: Silver, Gold and Platinum. Gold and Platinum members receive free late check-out until 2:00 PM and a room upgrade when available. Platinum members also receive one complimentary spa treatment per stay."),
        ("accessibility", "Accessibility", "Every Four Seasons property offers accessible rooms with roll-in showers, visual alarms, and step-free access to restaurants and the spa. Service animals are always welcome."),
    ],
}


def _fmt(t):  # "06:30" -> "6:30 AM"
    h, m = map(int, t.split(":"))
    return f"{(h % 12) or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def hotel_rows():
    return [dict(hotel_id=k, **v) for k, v in HOTELS.items()]


def kb_docs():
    docs = []
    for hid, items in _DOCS.items():
        hname = HOTELS[hid]["name"] if hid in HOTELS else BRAND
        docs += [dict(hotel_id=hid, category=c, title=t, body=b, hotel_name=hname) for c, t, b in items]
    # restaurant docs generated from the same records as the SQL table
    venues = defaultdict(list)
    for r in RESTAURANTS:
        venues[(r["hotel_id"], r["venue"])].append(r)
    for (hid, v), rows in venues.items():
        r0 = rows[0]
        hours = "; ".join(f"{r['meal_period']} {r['days']} {_fmt(r['open_time'])}-{_fmt(r['last_seating'])} (last seating/order)" for r in rows)
        body = (f"{v} - {r0['cuisine']}. Location: {r0['location']}. Hours: {hours}. Dress code: {r0['dress_code']}. "
                f"Reservations {'required' if r0['reservation_required'] else 'not required'}. "
                f"{'Children welcome.' if r0['kid_friendly'] else 'Adults-focused venue.'}")
        docs.append(dict(hotel_id=hid, category="dining", title=f"Restaurant: {v}", body=body, hotel_name=HOTELS[hid]["name"]))
    for hid, (spa, o, c, treatments) in SPA.items():
        cur = HOTELS[hid]["currency"]
        tr = "; ".join(f"{n} ({d} min, {cur} {p})" for n, d, p in treatments)
        docs.append(dict(hotel_id=hid, category="spa", title=f"Spa: {spa}", hotel_name=HOTELS[hid]["name"],
                         body=f"{spa} is open daily {_fmt(o)}-{_fmt(c)}. Treatments: {tr}. Book through the concierge; 24-hour cancellation."))
    for i, d in enumerate(docs):
        d["doc_id"] = f"{d['hotel_id']}-{d['category']}-{i:03d}"
        # the embedded text carries its own context (hotel + section) so chunks are self-describing
        d["content"] = f"{d['hotel_name']} | {d['title']}\n{d['body']}"
    return docs


def spa_slots(start: date | None = None, days: int = 14, seed: int = 7):
    rnd, start = random.Random(seed), start or date.today()
    rows = []
    for hid, (_, _, _, treatments) in SPA.items():
        for d in range(days):
            for t in ["10:00", "12:00", "14:00", "16:00", "18:00"]:
                for name, dur, price in treatments:
                    rows.append(dict(hotel_id=hid, slot_date=start + timedelta(days=d), slot_time=t, treatment=name,
                                     duration_min=dur, price=float(price), currency=HOTELS[hid]["currency"],
                                     is_available=rnd.random() > 0.4))
    return rows


def reservations(start: date | None = None):
    s = start or date.today()
    cols = ["confirmation_number", "first_name", "last_name", "email", "phone", "hotel_id", "arrival_date",
            "departure_date", "room_type", "adults", "children", "loyalty_tier", "preferences"]
    rows = [
        ("FS-48213", "Émilie", "Tremblay", "emilie.t@example.com", "+1-514-555-0142", "MTL", s + timedelta(1), s + timedelta(3), "Junior Suite, Old Port view", 2, 0, "Gold", "High floor; feather-free pillows"),
        ("FS-51907", "Daniel", "Okafor", "d.okafor@example.com", "+1-604-555-0199", "WHI", s + timedelta(3), s + timedelta(8), "Two-Bedroom Mountain Residence", 2, 2, "Platinum", "Kids club interest; ski valet"),
        ("FS-60344", "Sofía", "Marín", "sofia.marin@example.com", "+1-416-555-0110", "TOR", s, s + timedelta(4), "Deluxe Room, Yorkville view", 2, 0, "Silver", "Vegetarian; anniversary"),
    ]
    return [dict(zip(cols, r)) for r in rows]
