from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import requests
import xml.etree.ElementTree as ET
from urllib.parse import quote
import math
import os

app = Flask(__name__)
CORS(app)

NEWS_API_KEY         = "e1f66eb2925045388057984e26b66da8"

CREDIBLE_SOURCES = [
    "bbc", "reuters", "times of india", "ndtv", "the hindu",
    "hindustan times", "india today", "cnn", "the guardian",
    "associated press", "bloomberg", "economic times",
    "al jazeera", "washington post", "new york times",
    "abc news", "nbc news", "npr", "the wire", "scroll",
    "the print", "livemint", "business standard", "pib",
    "doordarshan", "ani", "pti", "dd news", "zee news",
    "republic world", "firstpost", "mint", "deccan herald",
    "telegraph", "indian express", "the quint", "news18",
    "wion", "moneycontrol", "financial express"
]

print("🔄 Loading ensemble model...")
ml_model   = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
print("✅ Model loaded!")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

fake_keywords = [
    "wake up", "share before", "gets deleted", "get deleted",
    "big pharma", "deep state", "mainstream media hiding",
    "they don't want", "you won't believe", "suppressed for decades",
    "cure cancer", "cure diabetes", "miracle cure", "secret cure",
    "doctors hate", "doctors don't want", "banned by government",
    "whistleblower leaks", "cover up", "staged by", "hoax exposed",
    "microchips in vaccine", "5g towers", "illuminati", "new world order",
    "sheeple", "before it gets deleted", "share this now",
    "anonymous insiders", "they are hiding", "exposed by insider",
    "mainstream media won't tell", "what they don't want you",
    "bombshell leaked", "shocking truth", "hidden agenda",
]

real_keywords = [
    "according to", "official statement", "press release",
    "confirmed by", "ministry of", "government of",
    "peer reviewed", "published in", "clinical trial",
    "supreme court", "high court", "parliament passed",
    "reserve bank", "election commission", "isro", "nasa",
    "world health organisation", "who confirmed", "united nations",
    "spokesperson said", "data released", "report published",
    "research shows", "study found", "scientists say",
    "university researchers", "journal of", "gazette notification",
    "budget allocation", "official figures", "census data",
    "rbi governor", "chief justice", "prime minister said",
    "cabinet approved", "lok sabha", "rajya sabha",
    "associated press", "reuters", "bbc reported",
    "according to data", "official report", "government data",
    "health ministry", "finance ministry", "defence ministry",
    "court ruling", "confirmed in a statement",
    "said in a press conference", "uk body", "maritime authority",
]

AI_WORDS = [
    "delve", "straightforward", "underscore", "notable", "notably",
    "crucial", "it's worth noting", "furthermore", "nevertheless",
    "in conclusion", "to summarize", "it is important to note",
    "plays a crucial role", "in the realm of", "comprehensive",
    "multifaceted", "nuanced", "pivotal", "leverage", "utilize",
    "facilitate", "robust", "seamlessly", "groundbreaking",
    "revolutionary", "paradigm", "synergy", "holistic",
    "commendable", "it is evident", "shed light", "in today's",
    "rest assured", "embark", "tapestry", "vibrant community",
    "testament to", "game changer", "game-changer", "landscape"
]


def check_facts(claim):
    result = {"found": False, "claim": None, "rating": None,
              "source": None, "source_url": None, "reviewed_by": None}

    if GOOGLE_FACTCHECK_KEY != "YOUR_GOOGLE_KEY_HERE":
        try:
            response = requests.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"query": quote(claim[:150]), "key": GOOGLE_FACTCHECK_KEY, "pageSize": 3},
                timeout=5
            )
            data = response.json()
            if "claims" in data and data["claims"]:
                top    = data["claims"][0]
                review = top.get("claimReview", [{}])[0]
                result.update({
                    "found": True, "claim": top.get("text", claim),
                    "rating": review.get("textualRating", "Checked"),
                    "source": review.get("publisher", {}).get("name", "Fact Checker"),
                    "source_url": review.get("url", ""),
                    "reviewed_by": review.get("publisher", {}).get("name", ""),
                })
                return result
        except Exception:
            pass

    try:
        query   = quote(f"fact check {claim[:80]}")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(rss_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        root  = ET.fromstring(response.content)
        items = root.findall(".//item")

        fact_check_sites = [
            "snopes", "politifact", "factcheck", "afp", "fullfact",
            "boomlive", "altnews", "vishvasnews", "factly", "newschecker",
            "thequint", "thelogicalindian", "indiatoday fact"
        ]

        for item in items[:8]:
            title  = item.findtext("title")  or ""
            link   = item.findtext("link")   or ""
            source = item.findtext("source") or ""
            if any(fc in (title + link + source).lower() for fc in fact_check_sites):
                result.update({
                    "found": True, "claim": claim, "rating": "Fact Checked",
                    "source": source or "Fact Check Source",
                    "source_url": link, "reviewed_by": source,
                })
                return result
    except Exception:
        pass

    return result


def find_original_story(text):
    result = {"found": False, "original_title": None, "original_source": None,
              "original_url": None, "similarity": None, "distortion_notes": []}
    try:
        query   = quote(' '.join(text.split()[:12]))
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(rss_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        root  = ET.fromstring(response.content)
        items = root.findall(".//item")

        for item in items[:5]:
            title  = (item.findtext("title")  or "").strip()
            link   = (item.findtext("link")   or "").strip()
            source = (item.findtext("source") or "").strip()

            is_credible = any(cs in source.lower() for cs in CREDIBLE_SOURCES)
            if not is_credible:
                is_credible = any(cs in link.lower() for cs in CREDIBLE_SOURCES)

            if is_credible and title:
                user_words     = set(clean_text(text).split())
                original_words = set(clean_text(title).split())
                common         = user_words & original_words
                similarity     = round(len(common) / max(len(user_words | original_words), 1) * 100)

                notes = []
                for word in ["shocking","bombshell","secret","exposed","breaking","urgent","miracle"]:
                    if word in text.lower() and word not in title.lower():
                        notes.append(f'Added sensational word: "{word}"')
                if len(text.split()) > len(title.split()) * 1.8:
                    notes.append("Submitted text significantly longer than original")

                result.update({
                    "found": True, "original_title": title,
                    "original_source": source, "original_url": link,
                    "similarity": similarity, "distortion_notes": notes[:3],
                })
                return result
    except Exception:
        pass
    return result


def detect_ai_text(text):
    if len(text.split()) < 10:
        return {"score": 50, "label": "Too short to analyze", "signals": []}

    signals = []
    score_points = 0
    max_points   = 0
    text_lower   = text.lower()

    ai_hits = [w for w in AI_WORDS if w in text_lower]
    max_points += 30
    if len(ai_hits) >= 3:
        score_points += 30
        signals.append(f"Uses {len(ai_hits)} AI-typical phrases: {', '.join(ai_hits[:3])}")
    elif len(ai_hits) >= 1:
        score_points += 15
        signals.append(f'Contains AI-typical word: "{ai_hits[0]}"')

    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if len(s.strip()) > 5]
    max_points += 20
    if len(sentences) >= 3:
        lengths = [len(s.split()) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        std_dev = math.sqrt(sum((l - avg_len) ** 2 for l in lengths) / len(lengths))
        if std_dev < 5 and avg_len > 8:
            score_points += 20
            signals.append("Very uniform sentence lengths (AI trait)")
        elif std_dev < 8:
            score_points += 10

    max_points += 15
    if text.count('!') == 0 and len(re.findall(r'\b[A-Z]{2,}\b', text)) <= 2:
        score_points += 15
        signals.append("No ALL CAPS or excessive exclamation marks (AI trait)")

    words        = re.findall(r'\b\w+\b', text_lower)
    unique_ratio = len(set(words)) / max(len(words), 1)
    max_points  += 15
    if unique_ratio > 0.75 and len(words) > 20:
        score_points += 15
        signals.append(f"High vocabulary diversity ({round(unique_ratio*100)}%) — AI trait")

    informal_words = ["gonna","wanna","gotta","kinda","tbh","lol","omg","yeah","nope","okay","guys","ain't","dunno"]
    informal_hits  = [w for w in informal_words if w in text_lower]
    max_points    += 20
    if not informal_hits and len(words) > 15:
        score_points += 20
        signals.append("Formal neutral tone with no informal language (AI trait)")

    ai_score = max(0, min(100, round((score_points / max(max_points, 1)) * 100)))
    label    = "Likely AI-Generated" if ai_score >= 70 else "Possibly AI-Generated" if ai_score >= 45 else "Likely Human-Written"

    return {"score": ai_score, "label": label, "signals": signals[:3]}


def search_google_news(headline):
    try:
        rss_url  = f"https://news.google.com/rss/search?q={quote(headline[:100])}&hl=en-IN&gl=IN&ceid=IN:en"
        response = requests.get(rss_url, timeout=6, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code != 200:
            return {"found": False, "searched": True}

        items = ET.fromstring(response.content).findall(".//item")
        if not items:
            return {"found": False, "searched": True}

        for item in items[:5]:
            title  = item.findtext("title")  or ""
            link   = item.findtext("link")   or ""
            source = item.findtext("source") or ""
            is_credible = any(cs in source.lower() for cs in CREDIBLE_SOURCES) or \
                          any(cs in link.lower() for cs in CREDIBLE_SOURCES)
            if is_credible:
                return {"found": True, "credible": True, "searched": True,
                        "source_name": source or "Verified Source",
                        "article_title": title, "article_url": link}

        first = items[0]
        return {"found": True, "credible": False, "searched": True,
                "source_name": first.findtext("source") or "Unknown",
                "article_title": first.findtext("title") or "",
                "article_url": first.findtext("link") or ""}
    except Exception:
        return {"found": False, "searched": False}


def search_newsapi(headline):
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": ' '.join(headline.split()[:10]), "apiKey": NEWS_API_KEY,
                    "pageSize": 5, "language": "en", "sortBy": "relevancy"},
            timeout=5
        )
        data = response.json()
        if data.get("status") != "ok" or not data.get("articles"):
            return {"found": False, "searched": True}

        for article in data["articles"]:
            sn = (article.get("source", {}).get("name") or "").lower()
            if any(cs in sn for cs in CREDIBLE_SOURCES):
                return {"found": True, "credible": True, "searched": True,
                        "source_name": article.get("source", {}).get("name"),
                        "article_title": article.get("title", ""),
                        "article_url": article.get("url", "")}

        first = data["articles"][0]
        return {"found": True, "credible": False, "searched": True,
                "source_name": first.get("source", {}).get("name", "Unknown"),
                "article_title": first.get("title", ""),
                "article_url": first.get("url", "")}
    except Exception:
        return {"found": False, "searched": False}


def search_news(headline):
    if NEWS_API_KEY != "YOUR_NEWSAPI_KEY_HERE":
        result = search_newsapi(headline)
        if result.get("found"):
            return result
    return search_google_news(headline)


def ml_predict(text):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    ml_pred = ml_model.predict(vec)[0]
    try:
        ml_prob = ml_model.predict_proba(vec)[0][1]
    except AttributeError:
        ml_prob = 0.8 if ml_pred == 1 else 0.2

    fake_hits = sum(1 for kw in fake_keywords if kw in cleaned)
    real_hits = sum(1 for kw in real_keywords if kw in cleaned)

    if len(re.findall(r'[A-Z]', text)) / max(len(text), 1) > 0.25: fake_hits += 2
    if text.count('!') >= 2: fake_hits += 1

    if fake_hits >= 2:
        return "FAKE", min(98, 78 + fake_hits * 4), ml_pred, fake_hits, real_hits
    elif real_hits >= 2:
        return "REAL", round(min(97, ((ml_prob * 0.5) + 0.5) * 100), 2), ml_pred, fake_hits, real_hits
    elif real_hits == 1 and ml_pred == 1:
        return "REAL", round(min(95, ml_prob * 100), 2), ml_pred, fake_hits, real_hits
    elif real_hits == 1 and ml_pred == 0:
        return "REAL", 63.0, ml_pred, fake_hits, real_hits
    else:
        if ml_prob >= 0.5:
            return "REAL", round(ml_prob * 100, 2), ml_pred, fake_hits, real_hits
        else:
            return "FAKE", round((1 - ml_prob) * 100, 2), ml_pred, fake_hits, real_hits


@app.route('/')
def home():
    return "🚀 The Centre Crew — Fake News Detection API is Live!"


@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    word_count = len(text.split())

    if word_count < 5:
        return jsonify({
            "label": "UNCERTAIN", "confidence": 50.0,
            "ml_prediction": "Too short", "bert_prediction": "N/A",
            "fake_signals": 0, "web_verified": False,
            "verified_by": None, "article_title": None, "article_url": None,
            "fact_check": {"found": False},
            "original_story": {"found": False},
            "ai_detection": {"score": 50, "label": "Too short", "signals": []},
            "message": "Too short to analyze. Please enter a full headline or article."
        })

    search_result  = search_news(text)
    fact_check     = check_facts(text)
    original_story = find_original_story(text)
    ai_detection   = detect_ai_text(text)

    ml_label, ml_confidence, ml_pred, fake_hits, real_hits = ml_predict(text)

    is_short_text = word_count <= 25
    web_searched  = search_result.get("searched", False)
    web_found     = search_result.get("found", False)
    web_credible  = search_result.get("credible", False)

    final_label = ml_label
    confidence  = ml_confidence
    verified_by = article_title = article_url = None

    if web_found and web_credible:
        final_label   = "REAL"
        confidence    = 97.0
        verified_by   = search_result.get("source_name")
        article_title = search_result.get("article_title")
        article_url   = search_result.get("article_url")
    elif web_found and not web_credible:
        verified_by   = search_result.get("source_name") + " (unverified)"
        article_title = search_result.get("article_title")
        article_url   = search_result.get("article_url")
    elif web_searched and not web_found:
        if is_short_text:
            if fake_hits >= 2:
                final_label = "FAKE"
                confidence  = min(98, 78 + fake_hits * 4)
            elif real_hits >= 2:
                final_label = "REAL"
                confidence  = 62.0
            else:
                final_label = "FAKE"
                confidence  = 72.0

    if fact_check.get("found") and final_label == "REAL":
        rating = (fact_check.get("rating") or "").lower()
        if any(w in rating for w in ["false","fake","misleading","wrong","incorrect","pants on fire"]):
            final_label = "FAKE"
            confidence  = 95.0

    return jsonify({
        "label":           final_label,
        "confidence":      round(confidence, 2),
        "ml_prediction":   "REAL" if ml_pred == 1 else "FAKE",
        "bert_prediction": "N/A",
        "fake_signals":    fake_hits,
        "web_verified":    web_found and web_credible,
        "web_searched":    web_searched,
        "verified_by":     verified_by,
        "article_title":   article_title,
        "article_url":     article_url,
        "fact_check":      fact_check,
        "original_story":  original_story,
        "ai_detection":    ai_detection,
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
