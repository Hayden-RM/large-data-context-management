"""
Generate a synthetic needle-in-a-haystack corpus for exp01.

Produces:
  data/samples/corpus.txt      — ~50k token document
  data/samples/needles.json    — ground truth Q&A pairs
"""

import json
import random
import textwrap
from pathlib import Path

SEED = 42
random.seed(SEED)

# --- Needles: discrete factual claims -------------------------------------------

NEEDLES: list[dict] = [
    {"id": f"needle_{i:02d}", "fact": fact, "question": question, "answer": answer}
    for i, (fact, question, answer) in enumerate([
        (
            "The melting point of gallium is 29.76 degrees Celsius.",
            "What is the melting point of gallium?",
            "29.76 degrees Celsius",
        ),
        (
            "Project Helix was launched on March 4, 2019.",
            "When was Project Helix launched?",
            "March 4, 2019",
        ),
        (
            "The Arundel Bridge has a load capacity of 42 tonnes.",
            "What is the load capacity of the Arundel Bridge?",
            "42 tonnes",
        ),
        (
            "Dr. Priya Nair holds the record for the fastest completion of the Vantara Trail at 6 hours and 11 minutes.",
            "Who holds the record for the Vantara Trail, and what is the time?",
            "Dr. Priya Nair, 6 hours and 11 minutes",
        ),
        (
            "The population of Kelwick as of the 2022 census was 14,830.",
            "What was the population of Kelwick in the 2022 census?",
            "14,830",
        ),
        (
            "Compound XR-7 inhibits enzyme CYP3A4 at a concentration of 0.3 micromolar.",
            "At what concentration does Compound XR-7 inhibit CYP3A4?",
            "0.3 micromolar",
        ),
        (
            "The Thornfield Protocol requires a 72-hour quarantine period.",
            "How long is the quarantine period under the Thornfield Protocol?",
            "72 hours",
        ),
        (
            "Server cluster DELTA-9 has a maximum throughput of 8.4 gigabits per second.",
            "What is the maximum throughput of server cluster DELTA-9?",
            "8.4 gigabits per second",
        ),
        (
            "The Verona Accord was signed by 14 nations on August 22, 1998.",
            "How many nations signed the Verona Accord, and on what date?",
            "14 nations, August 22, 1998",
        ),
        (
            "Lake Cavendish reaches a maximum depth of 187 metres.",
            "What is the maximum depth of Lake Cavendish?",
            "187 metres",
        ),
        (
            "The half-life of isotope RX-209 is 3.7 years.",
            "What is the half-life of isotope RX-209?",
            "3.7 years",
        ),
        (
            "Agent Codenamed SPARROW was decommissioned on November 1, 2003.",
            "When was Agent Codenamed SPARROW decommissioned?",
            "November 1, 2003",
        ),
        (
            "The Mira Station solar array produces 2.1 megawatts at peak output.",
            "What is the peak output of the Mira Station solar array?",
            "2.1 megawatts",
        ),
        (
            "The Aldenvale Manuscript contains exactly 3,412 surviving pages.",
            "How many pages does the Aldenvale Manuscript contain?",
            "3,412 pages",
        ),
        (
            "Flight corridor ECHO-7 has a ceiling altitude of 41,000 feet.",
            "What is the ceiling altitude of flight corridor ECHO-7?",
            "41,000 feet",
        ),
        (
            "The Cassini alloy has a tensile strength of 980 megapascals.",
            "What is the tensile strength of the Cassini alloy?",
            "980 megapascals",
        ),
        (
            "Regulation 14-B mandates a minimum buffer zone of 500 metres around protected wetlands.",
            "What buffer zone does Regulation 14-B mandate around protected wetlands?",
            "500 metres",
        ),
        (
            "The Orellan fauna stage lasted approximately 4.3 million years.",
            "How long did the Orellan fauna stage last?",
            "approximately 4.3 million years",
        ),
        (
            "Warehouse sector 7G has a storage capacity of 18,000 cubic metres.",
            "What is the storage capacity of warehouse sector 7G?",
            "18,000 cubic metres",
        ),
        (
            "The Harwick Comet completes one orbit every 217 years.",
            "How long does the Harwick Comet take to complete one orbit?",
            "217 years",
        ),
    ])
]


# --- Filler text generation ------------------------------------------------------

TOPICS = [
    "urban planning", "marine biology", "railway engineering", "textile manufacturing",
    "medieval cartography", "soil chemistry", "actuarial science", "glaciology",
    "numismatics", "seismology", "apiculture", "dendrochronology", "volcanology",
    "archival science", "hydraulic engineering", "mycology", "palaeobotany",
    "metrology", "speleology", "ethnomusicology",
]

SENTENCE_TEMPLATES = [
    "The study of {topic} has undergone significant transformation over the past several decades.",
    "Researchers in {topic} frequently debate the relative merits of competing methodological frameworks.",
    "Advances in {topic} have been driven largely by improvements in instrumentation and data collection.",
    "Practitioners of {topic} often emphasise the importance of longitudinal datasets.",
    "The intersection of {topic} and computational methods continues to yield promising results.",
    "Historical records pertaining to {topic} reveal patterns that were previously difficult to detect.",
    "Contemporary approaches to {topic} draw on insights from adjacent disciplines.",
    "Funding for {topic} research has fluctuated considerably in recent years.",
    "Graduate programmes specialising in {topic} remain competitive at leading universities.",
    "The peer review process in {topic} has attracted renewed scrutiny from the broader scientific community.",
    "Conferences dedicated to {topic} draw participants from over forty countries annually.",
    "The terminology used in {topic} can be a source of confusion for those entering the field.",
    "Case studies in {topic} demonstrate the value of interdisciplinary collaboration.",
    "Standardisation efforts in {topic} have proceeded more slowly than many had hoped.",
    "The role of fieldwork in {topic} is both celebrated and contested within the discipline.",
]


def make_paragraph(n_sentences: int = 6) -> str:
    topic = random.choice(TOPICS)
    sentences = [
        random.choice(SENTENCE_TEMPLATES).format(topic=topic)
        for _ in range(n_sentences)
    ]
    return " ".join(sentences)


def build_corpus(target_tokens: int = 50_000) -> str:
    """
    Build a corpus of filler paragraphs with needles scattered throughout.
    Needle positions are uniformly distributed across the document.
    """
    # Estimate: ~1 token per 4 chars, paragraph ~400 chars → ~100 tokens
    tokens_per_paragraph = 100
    total_paragraphs = target_tokens // tokens_per_paragraph
    needle_positions = sorted(random.sample(range(total_paragraphs), len(NEEDLES)))

    paragraphs = []
    needle_iter = iter(zip(needle_positions, NEEDLES))
    next_pos, next_needle = next(needle_iter)

    for i in range(total_paragraphs):
        if i == next_pos:
            paragraphs.append(next_needle["fact"])
            try:
                next_pos, next_needle = next(needle_iter)
            except StopIteration:
                next_pos = -1
        else:
            paragraphs.append(make_paragraph())

    return "\n\n".join(paragraphs)


# --- Write outputs ---------------------------------------------------------------

if __name__ == "__main__":
    out_dir = Path(__file__).parent

    print("Generating corpus...")
    corpus = build_corpus(target_tokens=50_000)
    corpus_path = out_dir / "corpus.txt"
    corpus_path.write_text(corpus)

    from sys import path as syspath
    import os
    # Allow importing src from repo root
    repo_root = Path(__file__).resolve().parents[2]
    syspath.insert(0, str(repo_root))

    from src.tokens import count_tokens
    token_count = count_tokens(corpus)
    print(f"Corpus written to {corpus_path} ({token_count:,} tokens, {len(corpus):,} chars)")

    needles_path = out_dir / "needles.json"
    needles_path.write_text(json.dumps(NEEDLES, indent=2))
    print(f"Needles written to {needles_path} ({len(NEEDLES)} Q&A pairs)")
