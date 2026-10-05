"""
Deterministic synthetic dataset generator for the Smart Text Classifier.

Produces data/texts.csv with ~560 customer-support style texts labelled
Complaint / Inquiry / Feedback / Other (~140 per class).

How it works:
- Hand-written seed examples (25 per class) define realistic phrasing.
- Each seed is expanded by template variation: synonym swaps, sentence
  reordering/casual rewrites, punctuation and casing changes, and a small
  chance of a short follow-up clause. All randomness is seeded so the CSV
  is byte-identical across runs (random + numpy seeds fixed below).
- A dedup pass guarantees no exact duplicates within the file.

This is synthetic data (see data/DATASET.md); it is designed to be realistic
but it is not collected from real users.
"""

from __future__ import annotations

import csv
import os
import random
import re
from pathlib import Path

SEED = 42
random.seed(SEED)

OUT = Path(__file__).resolve().parent.parent / "data" / "texts.csv"

# ---------------------------------------------------------------- seeds ------
# 25 hand-written seed sentences per class. {n} placeholders are filled with
# realistic order/product details.

SEEDS: dict[str, list[str]] = {
    "Complaint": [
        "My order arrived damaged and the box was completely crushed",
        "I want a refund for order #{n}, the item never arrived",
        "The delivery driver left my package in the rain and everything is ruined",
        "I have been charged twice for the same purchase, this is unacceptable",
        "Your app keeps crashing every time I try to check out",
        "The product I received is not what was advertised on the website",
        "I waited three weeks for my delivery and it still has not shipped",
        "The customer support agent was rude and hung up on me",
        "My subscription was renewed without my permission, cancel it now",
        "The shirt I bought shrank after one wash, I want my money back",
        "I received the wrong size even though I ordered {n}",
        "The website took my payment but my order confirmation never came",
        "This is the third time my package has been lost by your courier",
        "The item broke within two days of normal use, terrible quality",
        "I was promised a callback yesterday and nobody called me back",
        "My refund has been pending for over a month now",
        "The driver refused to deliver to my address and marked it delivered",
        "You sent me a used product instead of a new one",
        "The discount code on my invoice was not applied at checkout",
        "I am extremely disappointed with the quality of this purchase",
        "The tracking has not updated in ten days, where is my parcel",
        "My account was locked for no reason and I cannot log in",
        "The restaurant sent cold food and half the order was missing",
        "I cancelled within the free trial period and still got billed",
        "The headphones stopped working on the right side after a week",
    ],
    "Inquiry": [
        "What is the return policy for items bought during the sale",
        "How long does standard delivery take to {n}",
        "Do you offer international shipping to {n}",
        "Can I change the delivery address on order #{n}",
        "Is this laptop compatible with {n} operating systems",
        "What payment methods do you accept on your website",
        "How do I track my order once it has been dispatched",
        "Are there any discounts available for first time customers",
        "What is the warranty period on the {n} model",
        "Can I pay cash on delivery for orders over {n}",
        "Do you have this jacket in size {n}",
        "How do I cancel my subscription before the next billing cycle",
        "Is there a store near {n} where I can try this on",
        "What are your customer service hours on weekends",
        "Can I use two promo codes on a single order",
        "How much does express shipping cost to {n}",
        "Does this phone support dual SIM cards",
        "What is the difference between the basic and premium plans",
        "Can I get an invoice with my company name on it",
        "How do I reset my password if I forgot my email",
        "Do you restock items that are currently out of stock",
        "What documents do I need to open a return request",
        "Is installation included with the purchase of this appliance",
        "Can I upgrade my plan in the middle of the month",
        "How do I contact the seller directly about a custom order",
    ],
    "Feedback": [
        "The new checkout flow is so much faster, great job on the update",
        "I love the quality of the fabric, will definitely order again",
        "The support team resolved my issue within minutes, very impressed",
        "Delivery was earlier than expected and the packaging was excellent",
        "The app interface is clean and easy to navigate, nice work",
        "This is the best customer service I have experienced in years",
        "The tutorial videos really helped me set everything up quickly",
        "I appreciate how transparent your pricing is, no hidden fees",
        "The product exceeded my expectations, five stars from me",
        "Your team went above and beyond to help me today, thank you",
        "The new dark mode looks fantastic on my phone",
        "Shipping to {n} was surprisingly quick, well done",
        "I really like the new search filters, finding items is much easier",
        "The loyalty rewards program is a great idea, I feel valued",
        "Everything about this purchase was smooth from start to finish",
        "The installation technician was polite and professional",
        "I am happy with the free gift that came with my order",
        "The size guide on your site is accurate, the fit is perfect",
        "You have the friendliest delivery drivers in the business",
        "The refund process was painless and fast, I appreciate it",
        "The product manual is clear and well illustrated",
        "I love that you use eco friendly packaging for all orders",
        "The live chat support is a lifesaver, quick and helpful",
        "This update fixed all the bugs I reported last month",
        "I recommend your store to all my friends, keep it up",
    ],
    "Other": [
        "Just wanted to say hi, hope you are having a great day",
        "I read your latest blog post about {n}, interesting read",
        "Can you share the link to your careers page",
        "I am writing an article and would love a quote from your team",
        "Do you sponsor local community events in {n}",
        "Please add me to your newsletter mailing list",
        "I found your store through a friend's recommendation",
        "What is the story behind your brand name",
        "I am a student doing research on {n}, can you help",
        "Could you send me your press kit for a media feature",
        "I noticed a typo on your about us page, just letting you know",
        "Do you have an affiliate or referral program I can join",
        "I would like to partner with your brand for a giveaway",
        "Can I visit your warehouse for a school project tour",
        "I am trying to reach your marketing department",
        "Your Instagram reels are really entertaining, keep posting",
        "I lost the receipt from my in store purchase last week",
        "Do you offer gift wrapping for the holiday season",
        "I am moving to {n} next month, do you deliver there",
        "Can you tell me who founded the company and when",
        "I want to update my profile picture on my account",
        "Is there a way to download my order history as a file",
        "I accidentally created two accounts, can you merge them",
        "Do you have a mobile app for {n} devices",
        "I am curious about your sustainability practices",
    ],
}

# ------------------------------------------------------------ variation ------
_SYNONYMS: dict[str, list[str]] = {
    "damaged": ["broken", "smashed", "ruined"],
    "refund": ["money back", "reimbursement"],
    "quick": ["fast", "speedy"],
    "great": ["excellent", "amazing", "fantastic"],
    "terrible": ["awful", "horrible", "dreadful"],
    "rude": ["impolite", "disrespectful"],
    "fast": ["quick", "prompt"],
    "helpful": ["supportive", "useful"],
    "love": ["really like", "adore"],
    "hate": ["dislike", "can't stand"],
    "easy": ["simple", "straightforward"],
    "slow": ["sluggish", "delayed"],
    "expensive": ["pricey", "costly"],
    "cheap": ["affordable", "inexpensive"],
    "excellent": ["outstanding", "superb"],
    "disappointed": ["let down", "dissatisfied"],
    "happy": ["pleased", "glad"],
    "impressed": ["amazed", "delighted"],
    "polite": ["courteous", "friendly"],
    "quickly": ["fast", "promptly"],
}

_FOLLOWUPS: list[str] = [
    "Please look into this as soon as possible.",
    "Thanks in advance for your help.",
    "I would appreciate a quick response.",
    "Let me know if you need any more details.",
    "This is really important to me.",
    "Looking forward to hearing from you.",
    "I hope this can be sorted out soon.",
    "Thanks for your time.",
]

_DETAIL_VALUES: list[str] = [
    "#48213", "#90210", "#11754", "#66302", "Karachi", "Lahore",
    "Islamabad", "Dubai", "London", "size M", "size L", "size 42",
    "Windows and Mac", "Android", "iOS", "$200", "Rs 5000", "PKR 2500",
    "the Pro model", "the 2024 edition", "renewable energy", "AI tools",
]

_INTRO_OPENERS: list[str] = [
    "Hi, ", "Hello, ", "Hey, ", "Good morning, ", "Hi there, ", "",
]

_CLOSERS: list[str] = [
    ". Thanks", ". Thank you", "!", ".", "?", ". Regards",
]


def _synonym_swap(text: str) -> str:
    words = text.split()
    out = []
    for w in words:
        key = w.strip(".,!?").lower()
        if key in _SYNONYMS and random.random() < 0.35:
            choices = _SYNONYMS[key]
            swap = random.choice(choices)
            if w[0].isupper():
                swap = swap[0].upper() + swap[1:]
            out.append(swap)
        else:
            out.append(w)
    return " ".join(out)


def _fill_details(text: str) -> str:
    while "{n}" in text:
        text = text.replace("{n}", random.choice(_DETAIL_VALUES), 1)
    return text


def _vary_punctuation(text: str) -> str:
    if random.random() < 0.30:
        op = random.choice(_CLOSERS)
        text = re.sub(r"[.!?]+$", "", text) + op
    return text


def _vary_casing(text: str) -> str:
    r = random.random()
    if r < 0.10:
        return text.upper()
    if r < 0.20:
        return text[0].lower() + text[1:] if text else text
    return text


def _maybe_followup(text: str) -> str:
    if random.random() < 0.25:
        text = text.rstrip(".!?") + ". " + random.choice(_FOLLOWUPS)
    return text


def _maybe_opener(text: str) -> str:
    if random.random() < 0.25:
        text = random.choice(_INTRO_OPENERS) + text
    return text


def _shuffle_clauses(text: str) -> str:
    parts = re.split(r",\s*|\s+and\s+", text)
    if len(parts) >= 2 and random.random() < 0.25:
        random.shuffle(parts)
        return ", ".join(p.strip() for p in parts if p.strip())
    return text


def vary(seed_text: str) -> str:
    t = _fill_details(seed_text)
    t = _synonym_swap(t)
    t = _shuffle_clauses(t)
    t = _maybe_opener(t)
    t = _maybe_followup(t)
    t = _vary_punctuation(t)
    t = _vary_casing(t)
    return " ".join(t.split())


def main() -> None:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    per_class = 140

    for label, seeds in SEEDS.items():
        made = 0
        attempts = 0
        while made < per_class and attempts < per_class * 40:
            attempts += 1
            text = vary(random.choice(seeds))
            key = text.lower()
            if key in seen or len(text.split()) < 4:
                continue
            seen.add(key)
            rows.append((text, label))
            made += 1
        assert made == per_class, f"only generated {made} for {label}"

    random.shuffle(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")
    from collections import Counter
    print(Counter(label for _, label in rows))


if __name__ == "__main__":
    main()
