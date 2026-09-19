# conversation/language.py

import re

BENGALI_MARKERS = {
    # Exclusive (weight 2.0)
    "ache": 2.0, "aache": 2.0, "nei": 2.0, "hobe": 2.0, "hbe": 2.0,
    "koreche": 2.0, "korbo": 2.0, "jacchi": 2.0, "kothay": 2.0,
    "theke": 2.0, "jonno": 2.0, "ebong": 2.0, "somundro": 2.0,
    "kintu": 2.0, "bhalo": 2.0, "uchit": 2.0, "dekhaiche": 2.0,
    "dekhaicche": 2.0, "jekhane": 2.0, "tahole": 2.0, "naki": 2.0,
    "obosthay": 2.0, "jaygay": 2.0, "fire": 2.0, "konta": 2.0,
    "kemne": 2.0, "karon": 2.0, "kache": 2.0, "kachhe": 2.0,
    "dike": 2.0, "sange": 2.0, "brishti": 2.0,
    "ajke": 2.0, "ta": 1.0, "kemon": 2.0, "toh": 2.0, 
    "gele": 2.0, "ekhon": 2.0, "aar": 2.0, "na": 1.0, "jabo": 2.0,
    "mach": 2.0, "machh": 2.0,
    "kom": 2.0, "hole": 2.0, "krte": 2.0, "nirapod": 2.0, "er": 2.0,
    "samudre": 2.0, "korte": 2.0, "jani": 2.0,
    
    # Shared/ambiguous (weight 1.0)
    "ami": 1.0, "amar": 1.0, "amake": 1.0, "tumi": 1.0, "tomar": 1.0,
    "amader": 1.0, "tader": 1.0, "apni": 1.0, "apnar": 1.0,
    "jawa": 1.0, "jaowa": 1.0, "berobo": 1.0, "beriye": 1.0,
    "dekha": 1.0, "kelre": 1.0, "kon": 1.0, "keon": 1.0,
    "shokale": 1.0, "shokal": 1.0, "dupure": 1.0, "bikel": 1.0, "raat": 1.0,
    "hawa": 1.0, "baire": 1.0, "ki": 1.0,
    "khubi": 1.0, "safer": 1.0, "kam": 1.0, "temon": 1.0, "kaalke": 1.0
}

HINDI_MARKERS = {
    # Exclusive (weight 2.0)
    "hai": 2.0, "hain": 2.0, "ho": 2.0, "hoon": 2.0, "thi": 2.0, "tha": 2.0,
    "rahega": 2.0, "hoga": 2.0, "jaayein": 2.0, "karein": 2.0, "batayein": 2.0,
    "kahan": 2.0, "kaunsa": 2.0, "kaun": 2.0, "kya": 2.0, "kaise": 2.0,
    "kyun": 2.0, "agar": 2.0, "lekin": 2.0, "mein": 2.0, "ko": 2.0,
    "paas": 2.0, "ka": 2.0, "machli": 2.0, "samundar": 2.0,
    "subah": 2.0, "dopahar": 2.0, "shaam": 2.0, "baarish": 2.0, "sthiti": 2.0,
    "jagaah": 2.0, "wapas": 2.0, "accha": 2.0, "bahut": 2.0, "chahiye": 2.0,
    "zaroori": 2.0,
    "h": 2.0, "saf": 2.0, "sef": 2.0, "mn": 2.0, "kl": 2.0, "kaisa": 2.0,
    "subh": 2.0, "abhi": 2.0, "aur": 2.0, "dakshin": 2.0, "disha": 2.0,
    
    # Shared/ambiguous (weight 1.0)
    "main": 1.0, "mera": 1.0, "mere": 1.0, "meri": 1.0, "hum": 1.0,
    "humara": 1.0, "aap": 1.0, "tum": 1.0, "jaana": 1.0, "jaao": 1.0,
    "karne": 1.0, "karna": 1.0, "chalna": 1.0, "dekhein": 1.0, "par": 1.0,
    "se": 1.0, "tak": 1.0, "ke": 1.0, "ki": 1.0
}

def detect_language(query: str, prior_language: str = None) -> str:
    """Deterministic scored language detector for Romanized Indian languages (Banglish vs Hinglish vs English).
    Returns 'bn-Latn', 'hi-Latn', or 'en'.
    """
    if not query:
        return "en"
    
    words = re.findall(r"\b[a-z'-]+\b", query.lower())
    
    bn_score = 0
    hi_score = 0
    
    # Check for Bengali Unicode characters (\u0980-\u09FF)
    if re.search(r'[\u0980-\u09FF]', query):
        return "bn"
    
    for word in words:
        bn_score += BENGALI_MARKERS.get(word, 0)
        hi_score += HINDI_MARKERS.get(word, 0)

    if bn_score > 0 and bn_score >= hi_score:
        return "bn-Latn"
    elif hi_score > 0 and hi_score > bn_score:
        return "hi-Latn"
        
    if prior_language and prior_language in ("bn-Latn", "hi-Latn", "bn"):
        return prior_language
    
    return "en"
