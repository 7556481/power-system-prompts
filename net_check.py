import requests

tests = [
    ("Crossref", "https://api.crossref.org/works?rows=1"),
    ("SemanticScholar", "https://api.semanticscholar.org/graph/v1/paper/search?query=voltage%20stability&limit=1"),
    ("GitHub", "https://api.github.com"),
]

for name, url in tests:
    try:
        r = requests.get(url, timeout=10)
        print(name, r.status_code, r.text[:80].replace("\n"," "))
    except Exception as e:
        print(name, "FAILED:", repr(e))
