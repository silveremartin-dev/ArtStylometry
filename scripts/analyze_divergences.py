import json

with open("data/processed/contested_masters_audit.json", encoding="utf-8") as f:
    data = json.load(f)

type_a = []  # Official Canon -> Flagged Anomaly/Mouton
type_b = []  # Official Studio/Follower/Disputed -> High Master Score
consensus_canon = []
consensus_studio = []

for r in data:
    cat = r.get("attribution_category", "")
    verdict = r.get("verdict", "")
    conf = r.get("attribution_confidence", 0.0)
    is_bs = r.get("is_black_sheep", False)
    attr = r.get("official_attribution", "").lower()
    artist_canon = r.get("artist_canonical", "").lower()
    
    is_official_master = ("CANONICAL" in cat or cat == "CONSENSUS_MASTERWORK" or (attr == artist_canon))
    is_official_demoted = ("STUDIO" in cat or "CIRCLE" in cat or "FOLLOWER" in cat or "DISPUTED" in cat or 
                           "workshop" in attr or "circle" in attr or "follower" in attr or "attributed" in attr)
    
    if is_official_master and is_bs:
        type_a.append(r)
    elif is_official_demoted and (verdict == "CONSENSUS_MASTERWORK" or conf >= 0.70):
        type_b.append(r)
    elif is_official_master and not is_bs:
        consensus_canon.append(r)
    else:
        consensus_studio.append(r)

print(f"Total Evaluated: {len(data)}")
print(f"  [DIVERGENCE A] Official Canon -> Flagged Anomaly / Mouton Noir: {len(type_a)}")
print(f"  [DIVERGENCE B] Official Studio/Follower -> Rediscovered Autograph Affinity: {len(type_b)}")
print(f"  [CONSENSUS] Canonical Master Confirmed: {len(consensus_canon)}")
print(f"  [CONSENSUS] Studio / Secondary Hand Confirmed: {len(consensus_studio)}")

print("\n--- SAMPLE DIVERGENCE A (Flagged Canonical Artworks) ---")
for r in type_a[:6]:
    print(f" * [{r['artist_canonical']}] \"{r['title']}\" ({r['museum']})")
    print(f"   Official: {r['official_attribution']} -> Model: M-dist={r['mahalanobis_distance']:.2f}, Conf={r['attribution_confidence']*100:.0f}%, p={r['conformal_p_value']:.3f}")

print("\n--- SAMPLE DIVERGENCE B (Rediscovered Autograph Quality) ---")
for r in type_b[:6]:
    print(f" * [{r['artist_canonical']}] \"{r['title']}\" ({r['museum']})")
    print(f"   Official: {r['official_attribution']} -> Model: M-dist={r['mahalanobis_distance']:.2f}, Conf={r['attribution_confidence']*100:.0f}%, p={r['conformal_p_value']:.3f}")
