import itertools
import random
import json
import re

HINGLISH_SLOTS = {
  "time": [
    ("kal subah", {"relative": "tomorrow", "period": "morning"}),
    ("kal", {"relative": "tomorrow", "period": "none"}),
    ("tomorrow morning", {"relative": "tomorrow", "period": "morning"}),
    ("tmrw", {"relative": "tomorrow", "period": "none"}),
    ("kal shokl", {"relative": "tomorrow", "period": "morning"})
  ],
  "location": [
    ("kochi ke paas", "Kochi"),
    ("kochi area mein", "Kochi"),
    ("near kochi", "Kochi"),
    ("mumbai coast par", "Mumbai"),
    ("goa ke paas", "Goa"),
    ("near chennai", "Chennai"),
    ("vizag sea mein", "Vizag"),
    ("puri-te", "Puri"),
    ("randomtown mein", "UNKNOWN_LOC"),
    ("kisi jagah", "UNKNOWN_LOC"),
    ("unknown beach par", "UNKNOWN_LOC")
  ],
  "activity": [
    ("fishing", "fishing"),
    ("machli pakadne", "fishing"),
    ("sea mein", "none")
  ],
  "intent": [
    ("jaana safe hai?", "marine_safety"),
    ("safe rahega?", "marine_safety"),
    ("kaisa hai?", "marine_conditions"),
    ("pfz kaunsa hai?", "pfz_search"),
    ("nearest pfz?", "pfz_search"),
    ("cyclone alert hai?", "hazard_alert"),
    ("lightning alert", "hazard_alert")
  ]
}

BENGLISH_SLOTS = {
  "time": [
    ("kal shokale", {"relative": "tomorrow", "period": "morning"}),
    ("agamikal", {"relative": "tomorrow", "period": "none"}),
    ("tomorrow morning", {"relative": "tomorrow", "period": "morning"}),
    ("kal shkl", {"relative": "tomorrow", "period": "morning"}),
    ("kal shokl", {"relative": "tomorrow", "period": "morning"})
  ],
  "location": [
    ("kochi-r kache", "Kochi"),
    ("kochi-te", "Kochi"),
    ("kochi area-y", "Kochi"),
    ("digha-r kache", "Digha"),
    ("cox's bazar e", "Coxs_Bazar"),
    ("sundarban area-y", "Sundarbans"),
    ("puri-te", "Puri"),
    ("randomtown e", "UNKNOWN_LOC"),
    ("kothao ekta", "UNKNOWN_LOC"),
    ("unknown sea te", "UNKNOWN_LOC")
  ],
  "activity": [
    ("fishing korte", "fishing"),
    ("mach dhorte", "fishing"),
    ("samudre", "none")
  ],
  "intent": [
    ("jawa safe ki?", "marine_safety"),
    ("nirapod hobe?", "marine_safety"),
    ("kemon hobe?", "marine_conditions"),
    ("nearest pfz konta?", "pfz_search"),
    ("cyclone alert ache?", "hazard_alert")
  ]
}

OOD_EXAMPLES = [
    "What time is it?",
    "hello how are you",
    "aaj khane mein kya hai?",
    "tell me a joke",
    "cricket score kya hai",
    "ami valo achi",
    "where is the nearest hospital",
    "can you write an email for me"
]

ENGLISH_SLOTS = {
  "time": [
    ("tomorrow", {"relative": "tomorrow", "period": "none"}),
    ("tomorrow morning", {"relative": "tomorrow", "period": "morning"}),
    ("today", {"relative": "today", "period": "none"})
  ],
  "location": [
    ("near kochi", "Kochi"),
    ("around chennai", "Chennai"),
    ("off the coast of mumbai", "Mumbai"),
    ("in the middle of nowhere", "UNKNOWN_LOC")
  ],
  "activity": [
    ("to fish", "fishing"),
    ("for fishing", "fishing"),
    ("to go out", "none")
  ],
  "intent": [
    ("is it safe", "marine_safety"),
    ("how is the weather", "marine_conditions"),
    ("any cyclone alerts", "hazard_alert"),
    ("where is the nearest pfz", "pfz_search")
  ]
}

def drop_vowels(word):
  if len(word) <= 3: return word
  return word[0] + re.sub(r'[aeiou]', '', word[1:])

def phonetic_swap(word):
  swaps = {
    "safe": ["sef", "saf", "sf"],
    "fishing": ["fising", "phishing", "fshng", "fisshing"],
    "hai": ["h", "he"],
    "korte": ["krte", "korbo"],
    "shokale": ["shokl", "sokale", "shkl"],
    "morning": ["mrnng", "mrng"],
    "tomorrow": ["tmrrw", "tomrw", "tmrw"],
    "machli": ["mchl", "machhli"],
    "pakadne": ["pkdn", "pakadna"],
    "jaana": ["jn", "jana"]
  }
  if word.lower() in swaps:
    return random.choice(swaps[word.lower()])
  return word.replace('sh', 's').replace('z', 'j')

def inject_oov_noise(sentence, oov_prob=0.15):
  if random.random() > oov_prob:
      return sentence
  
  oov_tokens = ["😭", "🌊", "🛥️", "???", "..", "!!", "hmm"]
  token = random.choice(oov_tokens)
  
  if random.random() < 0.5:
      return sentence + " " + token
  else:
      words = sentence.split()
      insert_pos = random.randint(0, len(words))
      words.insert(insert_pos, token)
      return " ".join(words)

def apply_noise_and_shuffle(components, noise_prob=0.3, shuffle_prob=0.4):
  if random.random() < shuffle_prob:
      random.shuffle(components)
      
  sentence = " ".join(components)
  words = sentence.split()
  noisy_words = []
  
  for i, word in enumerate(words):
    if random.random() < noise_prob:
      mutation = random.choice([drop_vowels, phonetic_swap])
      word = mutation(word)
        
    if i < len(words) - 1 and random.random() < 0.10:
        word = word + words.pop(i+1)
        
    noisy_words.append(word)
      
  base_sentence = " ".join(noisy_words)
  return inject_oov_noise(base_sentence)

def generate_dataset(slots, lang_code, num_noise_variants=5):
  dataset = []
  combinations = list(itertools.product(
    slots["time"], slots["location"], slots["activity"], slots["intent"]
  ))
  
  for time_tup, loc_tup, act_tup, intent_tup in combinations:
    if act_tup[1] == "none" and random.random() < 0.5:
      components = [time_tup[0], loc_tup[0], intent_tup[0]]
    else:
      components = [time_tup[0], loc_tup[0], act_tup[0], intent_tup[0]]

    clean_sentence = " ".join(components)
    
    target_data = {
      "language_label": lang_code,
      "intent_label": intent_tup[1],
      "location_label": loc_tup[1],
      "activity_label": act_tup[1],
      "time_rel_label": time_tup[1]["relative"],
      "time_per_label": time_tup[1]["period"]
    }

    dataset.append({
        "input": clean_sentence,
        **target_data
    })
    
    for _ in range(num_noise_variants):
      dataset.append({
        "input": apply_noise_and_shuffle(components.copy()),
        **target_data
      })
          
  return dataset

if __name__ == "__main__":
  hinglish_data = generate_dataset(HINGLISH_SLOTS, "hi-Latn")
  benglish_data = generate_dataset(BENGLISH_SLOTS, "bn-Latn")
  english_data = generate_dataset(ENGLISH_SLOTS, "en")
  
  full_dataset = hinglish_data + benglish_data + english_data
  
  for ood_text in OOD_EXAMPLES:
    full_dataset.append({
      "input": ood_text,
      "language_label": "en",
      "intent_label": "unknown",
      "location_label": "UNKNOWN_LOC",
      "activity_label": "none",
      "time_rel_label": "none",
      "time_per_label": "none"
    })
      
  random.shuffle(full_dataset)
  
  with open("orca_training_encoder_data.jsonl", "w", encoding="utf-8") as f:
    for item in full_dataset:
      f.write(json.dumps(item) + "\n")
          
  print(f"Generated {len(full_dataset)} training examples.")