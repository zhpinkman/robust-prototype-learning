prompts = {
    "imdb": """Identify the binary sentiment of the following text:\n{text}.\n\nStrictly output only "negative" or "positive" according to the sentiment and nothing else.\nAssistant: """,
    "dbpedia": """Categorize the following text article strictly into only one taxonomic category from the following list: Agent, Work, Place, Species, UnitOfWork, Event, SportsSeason, Device, and TopicalConcept. Ensure that you output only the category name and nothing else.\n\nText: {text}\nAssistant: """,
    "ag_news": """Categorize the following news strictly into only one of the following classes: world, sports, business and science. Ensure that you output only the category name and nothing else.\n\nText: {text}\nAssistant: """,
}
