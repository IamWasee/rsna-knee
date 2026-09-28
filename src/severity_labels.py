"""Apply the host's severity thresholds to a report-derived label table.

The gold labels are severity- and acuity-thresholded (host Overview, thread
733343, quoted in research.md section 1.2): effusion and Baker's count only when
moderate or large; ACL and MCL only when high-grade or complete; OA only with
high-grade cartilage loss; fracture only when acute; contusion only without a
fracture; and "ambiguous or borderline findings ('on the fence') were graded as
negative". The table every arm trains on marks a finding present whenever the
report names it -- our own extractor's prompt said "a small effusion ... counts as
present" -- and an audit against the 58 found precision 0.69.

This keeps the table and moves only cells it marks present. For each one it finds
the report sentences that name that finding and reads their qualifiers:

  * a strong qualifier (moderate, large, complete, high-grade, grade 3-4, ...)
    anywhere keeps the cell;
  * otherwise a weak qualifier that the host's criteria make negative FOR THAT
    FINDING moves it to WEAK -- size and grade only where the host thresholds
    size (effusion, Baker's, ACL, MCL, OA), ligament-quality words only for the
    ACL and MCL, age only for fracture and MCL, and hedging for everything;
  * with no sentence found, or no qualifier, the cell is left alone.

WEAK sits below 0.5, so the finding binarises negative as gold grades it, but
above the table's "not mentioned" level (0.25), so a small effusion still ranks
above no word about effusion at all.

The word lists are written from the host's criteria and ordinary radiology
usage in the nine report languages, and their scoping was corrected by reading
rule decisions on reports OUTSIDE the 58. But the idea itself is not independent
of the 58: it came from reading the 50 gold reports whose labels were wrong,
before any rule existed. So the table-level gain measured on the 58 (precision
0.69 -> 0.80) is partly in-sample. The rules were then committed (2af829f) and
scored once, and must not be tuned against the 58 again. Run it, then score with
scripts/label_audit.py.

    python src/severity_labels.py data/audit/train.csv data/audit/llm_labels_v4_blend.csv \\
        data/audit/labels_severity.csv
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ID_COL, LABELS  # noqa: E402

WEAK = 0.35


def norm(text: str) -> str:
    """Lowercase, strip accents, and fold the letters NFD leaves alone.

    Greek reports here are typed with the micro sign (U+00B5) for mu, Turkish
    has a dotless i, German an eszett. Folding all of them lets one spelling of
    each keyword match every variant.
    """
    t = str(text).replace("µ", "μ").replace("ß", "ss")
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return t.replace("ı", "i").replace("ς", "σ")


# Sentence or clause breaks: full stops (not decimals), semicolons, newlines and
# the bullet markers these reports use ("- ", "> ", "* ", "1. ").
SPLIT = re.compile(r"(?<!\d)\.(?!\d)|;|\n|(?:^|\s)[->*•]\s|\s\d+\.\s")


def sentences(report: str) -> list[str]:
    return [s.strip() for s in SPLIT.split(norm(report)) if s and s.strip()]


def rx(words: list[str]) -> re.Pattern:
    return re.compile("|".join(words))


# ---- what names each finding, per language ---------------------------------
ANCHOR = {
    "Effusion": rx([r"effusion", r"hydrops", r"hemarthros", r"derrame", r"liquido articular",
                    r"efuzyon", r"eklem\w* (ici|icerisinde) \w*\s?sivi", r"sivi artis",
                    r"izljev", r"συλλογ\w* υγρου", r"υγρ\w* ενδαρθρ", r"εν[δθ]αρθρ\w* συλλογ", r"erguss",
                    r"излив", r"vocht", r"effusie", r"epanchement", r"joint fluid"]),
    "Baker's": rx([r"baker", r"popliteal cyst", r"quiste popliteo", r"popliteal kist",
                   r"poplitealn\w* cist", r"бейкър", r"popliteale cyste",
                   r"kyste poplite", r"poplitealzyste"]),
    "ACL": rx([r"\bacl\b", r"anterior cruciate", r"\blca\b", r"ligamento cruzado anterior",
               r"on capraz", r"prednj\w* kriz", r"προσθι\w* χιαστ", r"\bvkb\b",
               r"vordere\w* kreuzband", r"предн\w* кръстн", r"voorste kruisband",
               r"croise anterieur"]),
    "MCL": rx([r"\bmcl\b", r"medial collateral", r"\blcm\b", r"colateral medial",
               r"medial kollateral", r"medijaln\w* kolateral", r"εσω πλαγι",
               r"innenband", r"медиалн\w* колатерал", r"mediale collaterale",
               r"collateral interne", r"\blli\b"]),
    "Medial Meniscus": rx([r"medial menisc", r"menisco (medial|interno)", r"medial menisk",
                           r"medijaln\w* menisk", r"εσω μηνισκ", r"innenmenisk",
                           r"медиал\w* мениск", r"mediale menisc", r"menisque (interne|medial)"]),
    "Lateral Meniscus": rx([r"lateral menisc", r"menisco (lateral|externo)", r"lateral menisk",
                            r"lateraln\w* menisk", r"εξω μηνισκ", r"aussenmenisk",
                            r"латерал\w* мениск", r"laterale menisc",
                            r"menisque (externe|lateral)"]),
    "Synovitis": rx([r"synovit", r"sinovit", r"синовит", r"υμενιτ", r"συνοβιτ",
                     r"synovialit"]),
    "Contusion": rx([r"contusi", r"bruise", r"kontuzyon", r"kontuzi", r"контузи",
                     r"μωλωπ", r"botcontusie", r"bone marrow (o)?edema", r"edema oseo",
                     r"kemik iligi odem", r"kostan\w* edem", r"οστεομυελικ\w* οιδημ",
                     r"knochenodem", r"костно-?мозъч\w* едем", r"botoedeem",
                     r"oedeme osseux"]),
    "Fracture": rx([r"fractur", r"frakt", r"κατα[γκ]μ", r"фрактур", r"fractuur",
                    r"kiri[gk]", r"prelom"]),
}
# OA: a cartilage or arthrosis term together with the compartment.
# "(?<!sub)": subchondral bone is not cartilage.
CARTILAGE = rx([r"cartilag", r"(?<!sub)chondr", r"(?<!sub)condr", r"kikirdak", r"hrskavic", r"χονδρ",
                r"knorpel", r"хрущял", r"kraakbeen", r"osteoarthr", r"artrosis",
                r"gonartro", r"artroz", r"\boa\b", r"οστεοαρθρ", r"arthrose",
                r"артроз", r"artrotsk", r"hondromalac"])
COMPARTMENT = {
    "Medial OA": rx([r"medial", r"interno", r"εσω", r"медиал", r"mediaal", r"innen",
                     r"medijal"]),
    "Lateral OA": rx([r"lateral", r"externo", r"εξω", r"латерал", r"aussen", r"lateraal"]),
    "PF OA": rx([r"patel", r"rotul", r"trocl", r"trochl", r"επιγονατ", r"пател",
                 r"femoropatell", r"patelofemor", r"patellofemor"]),
}

# ---- qualifiers --------------------------------------------------------------
STRONG = rx([
    # en
    r"moderate", r"\blarge", r"massive", r"marked", r"severe", r"extensive",
    r"complete", r"full[- ]thickness", r"high[- ]grade", r"advanced", r"prominent",
    r"\btotal", r"rupture",
    # grades 3-4 in every notation used
    r"grad[eo]?\s*(iii|iv|[34])\b", r"\b(iii|iv|[34])\s*°", r"\b(iii|iv|[34])\.?\s*stupnj",
    r"icrs\s*(grade\s*)?(iii|iv|[34])\b",
    # es
    r"moderad", r"abundante", r"importante", r"sever[oa]", r"extens[oa]", r"complet[oa]",
    r"espesor total", r"masiv",
    # tr
    r"belirgin", r"orta derece", r"orta miktar", r"genis", r"yaygin", r"ileri", r"siddetli", r"komplet",
    r"tam kat", r"masif",
    # hr
    r"opsez", r"velik", r"umjeren", r"izrazen", r"tesk", r"kompletn", r"ruptur",
    # el
    r"μετρι", r"μεγαλ", r"ευμεγεθ", r"εκτεταμεν", r"πληρ\w*", r"ικανη ποσοτητα", r"ρηξ",
    # de
    r"massig", r"deutlich", r"ausgepragt", r"\bgross", r"erheblich", r"komplett",
    r"vollstandig", r"hochgradig",
    # bg
    r"умерен", r"голям", r"изразен", r"масивен", r"пълн", r"руптур",
    # nl
    r"matig", r"\bgroot", r"\bgrote", r"uitgesproken", r"ernstig", r"volledig", r"gevorderd",
    r"uitgebreid",
    # fr
    r"modere", r"abondant", r"volumineu", r"massif", r"transfixiant",
])
# Size and grade: the host thresholds these five -- moderate or large effusion and
# Baker's, high-grade ACL and MCL, high-grade cartilage loss for OA.
SIZE = rx([
    r"\bsmall", r"\bmild", r"minimal", r"\btrace", r"\btiny", r"\bslight", r"\bsome\b",
    r"low[- ]grade", r"\bearly\b", r"incipient", r"superficial",
    r"grad[eo]?\s*(ii|i|[12])\b", r"\b(ii|i|[12])\s*°", r"\b(ii|i|[12])\.?\s*stupnj",
    r"icrs\s*(grade\s*)?(ii|i|[12])\b", r"first degree",
    r"pequen", r"\bleve\b", r"minim[oa]", r"discret[oa]", r"escas[oa]", r"liger[oa]",
    r"incipiente",
    r"hafif", r"\baz\b", r"kucuk", r"erken", r"baslangic",
    r"\bmanj[aie]\b", r"blag", r"\bmal[aio]\b", r"pocetn",
    r"ηπι", r"μικρ", r"ελαχιστ", r"αρχομεν",
    r"gering", r"\bklein", r"\bleicht", r"diskret", r"\bspur\b", r"\bzart", r"beginnend",
    r"минимал", r"малк", r"малък", r"\bлек", r"незначител", r"начал",
    r"\bkleine?\b", r"\blichte?\b", r"minimaal", r"discreet",
    r"\bpetite?\b", r"minime", r"\blegere?\b", r"\bfaible", r"debutant",
])
SIZE_LABELS = {"Effusion", "Baker's", "ACL", "MCL", "Medial OA", "Lateral OA", "PF OA"}

# Ligament quality: "mild signal change, degeneration or thickening without
# discontinuity" is negative for the ACL, low-grade sprain for the MCL.
LIGAMENT = rx([r"interstitial", r"intrasubstance", r"mucoid", r"myxoid", r"sprain",
               r"\bstrain", r"distorsi", r"esguince", r"burkul", r"istegnut", r"θλασ",
               r"zerrung", r"разтягане", r"verstuiking", r"entorse"])
LIGAMENT_LABELS = {"ACL", "MCL"}

# Hedging: "on the fence" is negative, for every finding.
HEDGE = rx([
    r"possib", r"suspect", r"suspicious", r"suggest", r"\bmay\b", r"\bmight\b", r"could",
    r"cannot be excluded", r"can't be excluded", r"\br/o\b", r"rule out", r"equivocal",
    r"questionable", r"borderline",
    r"posible", r"podria", r"sospech", r"sugier", r"sugestiv", r"no se descarta",
    r"olasi", r"supheli", r"dusundur",
    r"moguc", r"moze odgovarat", r"sumnj",
    r"πιθαν", r"υποψι",
    r"moglich", r"verdacht", r"fraglich", r"\bdd\b",
    r"възможн", r"съмнени",
    r"mogelijk", r"eventueel",
    r"evoqu", r"suspicion", r"douteu",
])


# Acuity: the host grades only ACUTE fractures and ACUTE MCL tears positive;
# "chronic or remote stress changes are negative". Used for those two alone --
# "chronic synovitis" or "old ACL tear" say nothing against those labels.
ACUITY = rx([
    r"\bold\b", r"healed", r"chronic", r"remote", r"stress reaction", r"sequel",
    r"antigu", r"cronic", r"secuela", r"eski", r"kronik", r"stara", r"kronic",
    r"παλαι", r"χρονι", r"\balte\b", r"chronisch", r"стар", r"хроничн", r"\boude\b",
    r"ancien",
])
ACUITY_LABELS = {"Fracture", "MCL"}

# A strong word under negation says nothing: "no significant effusion", "sin
# rotura completa", "bez znakova rupture". Drop the negator and the three words
# after it before looking for strong qualifiers.
NEG = re.compile(r"\b(no|not|without|sin|bez|nema|yok|degil|δεν|χωρισ|без|няма|nicht|"
                 r"kein\w*|geen|niet|sans|ohne|pas de)\b(\s+\S+){1,3}")


def finding_sentences(label: str, sents: list[str]) -> list[str]:
    if label in COMPARTMENT:
        return [s for s in sents if CARTILAGE.search(s) and COMPARTMENT[label].search(s)]
    return [s for s in sents if ANCHOR[label].search(s)]


def verdict(label: str, sents: list[str]) -> tuple[str, str]:
    """'strong' | 'weak' | 'none', and the sentence that decided it."""
    hits = finding_sentences(label, sents)
    for s in hits:
        if STRONG.search(NEG.sub(" ", s)):
            return "strong", s
    for s in hits:
        if (HEDGE.search(s)
                or (label in SIZE_LABELS and SIZE.search(s))
                or (label in LIGAMENT_LABELS and LIGAMENT.search(s))
                or (label in ACUITY_LABELS and ACUITY.search(s))):
            return "weak", s
    return "none", hits[0] if hits else ""


def apply(train: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rep = train.set_index(ID_COL)["Report"]
    out = labels.copy().set_index(ID_COL)
    log = []
    for sid in out.index:
        if sid not in rep.index:
            continue
        sents = sentences(rep[sid])
        for c in LABELS:
            if out.at[sid, c] <= 0.5:
                continue
            v, s = verdict(c, sents)
            if v == "weak":
                log.append((sid, c, float(out.at[sid, c]), s))
                out.at[sid, c] = WEAK
        # Contusion is defined against fracture: oedema WITHOUT a fracture line.
        if out.at[sid, "Fracture"] > 0.5 and out.at[sid, "Contusion"] > 0.5:
            log.append((sid, "Contusion", float(out.at[sid, "Contusion"]),
                        "fracture present -> contusion negative by definition"))
            out.at[sid, "Contusion"] = WEAK
    return out.reset_index(), pd.DataFrame(log, columns=[ID_COL, "label", "was", "sentence"])


def main(train_csv: str, labels_csv: str, out_csv: str) -> None:
    train = pd.read_csv(train_csv)
    labels = pd.read_csv(labels_csv)
    new, log = apply(train, labels)
    new.to_csv(out_csv, index=False)
    log.to_csv(Path(out_csv).with_suffix(".log.csv"), index=False)
    before = (labels[LABELS] > 0.5).sum()
    after = (new[LABELS] > 0.5).sum()
    print(f"{'label':<17}{'present before':>15}{'after':>8}{'moved':>7}")
    for c in LABELS:
        print(f"{c:<17}{before[c]:>15}{after[c]:>8}{before[c] - after[c]:>7}")
    print(f"\n{len(log)} cells moved to {WEAK}; wrote {out_csv} and its .log.csv")


if __name__ == "__main__":
    main(*sys.argv[1:4])
