"""Finance-aware query understanding.

Analysts ask about "capex", "PP&E" or "the quick ratio", but a 10-K talks about
"purchases of property, plant and equipment" and "total current liabilities".
This module closes that vocabulary gap in three ways:

1. strips answer-formatting instructions ("round to two decimals") that add noise
2. expands finance shorthand into the line items a filing actually uses
3. works out which financial statements the question needs, so statement pages
   (balance sheet, income statement, cash flow statement) can be boosted
"""

import re

STATEMENT_HEADERS = {
    "cash_flow": r"statements? of (?:consolidated )?cash flows?|cash flows? statements?",
    "balance": r"balance sheets?|statements? of (?:consolidated )?financial (?:position|condition)",
    "income": r"statements? of (?:consolidated )?(?:operations|income|earnings)|income statements?",
}

# Phrases in a question that point at a statement directly
STATEMENT_MENTIONS = {
    "cash_flow": r"cash ?flow statement|statement of cash flows?",
    "balance": r"balance sheet|statement of financial position",
    "income": r"income statement|statement of (?:income|operations|earnings)|p ?& ?l\b|profit and loss",
}

# Finance shorthand -> (line-item vocabulary to add, statements it lives on)
GLOSSARY = [
    (r"\bcapex\b|capital expenditure", "purchases of property plant and equipment capital expenditures additions", {"cash_flow"}),
    (r"\bpp ?& ?e\b|\bppne\b|fixed asset", "property plant and equipment net", {"balance"}),
    (r"operating cash flow|cash from operations|cash flow from operations",
     "net cash provided by operating activities", {"cash_flow"}),
    (r"free cash ?flow", "net cash provided by operating activities purchases of property plant and equipment", {"cash_flow"}),
    (r"dividend", "dividends paid cash dividends declared", {"cash_flow"}),
    (r"\brevenue|top ?line|\bsales\b", "revenue revenues net sales total net revenue", {"income"}),
    (r"cogs|cost of goods", "cost of sales cost of goods sold cost of revenue", {"income"}),
    (r"gross (?:margin|profit)", "gross profit gross margin cost of sales", {"income"}),
    (r"operating (?:income|margin|profit)|\bebit\b", "operating income income from operations", {"income"}),
    (r"ebitda", "operating income depreciation and amortization", {"income", "cash_flow"}),
    (r"net income|net profit|net earnings|net margin", "net income net earnings attributable", {"income"}),
    (r"\beps\b|earnings per share", "earnings per share diluted basic", {"income"}),
    (r"effective tax|tax rate", "provision for income taxes income before income taxes", {"income"}),
    (r"interest expense|interest coverage", "interest expense", {"income"}),
    (r"current ratio|quick ratio|working capital|liquidity",
     "total current assets total current liabilities cash and cash equivalents receivables", {"balance"}),
    (r"inventor", "inventories merchandise inventories cost of sales", {"balance", "income"}),
    (r"payable", "accounts payable cost of sales", {"balance", "income"}),
    (r"receivable", "accounts receivable net", {"balance"}),
    (r"total assets|return on assets|\broa\b|asset turnover", "total assets net income", {"balance", "income"}),
    (r"equity|return on equity|\broe\b", "total shareholders equity stockholders equity", {"balance"}),
    (r"debt|leverage", "long term debt short term borrowings total debt", {"balance"}),
]

INSTRUCTION_PATTERNS = [
    r"round your answer[^.?]*[.?]",
    r"answer in (?:usd|units)[^.?]*[.?]",
    r"(?:assume|imagine) (?:that )?you are[^.:?]*[.:]",
    r"you are an? [a-z ]+(?:analyst|banker|investor)[^.:?]*[.:]",
    r"(?:give|provide) (?:a )?(?:response|solution|answer)[^.?]*[.?]",
    r"(?:please )?(?:base your|utilize|use) [^.?]*(?:information|details|line items)[^.?]*[.?]",
    r"calculate (?:what was asked|\(or extract\))[^.?]*[.?]",
    r"respond to the question[^.?]*[.?]",
    r"here'?s the question:",
]


def page_statements(page_text):
    """Which financial statements does this page look like? Checks the page header."""
    head = re.sub(r"\s+", " ", page_text[:400].lower())
    return {name for name, pattern in STATEMENT_HEADERS.items() if re.search(pattern, head)}


def analyze(question):
    """Return (search_query, statements_needed) for a question."""
    q = question.lower()
    statements = {name for name, pattern in STATEMENT_MENTIONS.items() if re.search(pattern, q)}

    core = q
    for pattern in INSTRUCTION_PATTERNS:
        core = re.sub(pattern, " ", core)

    expansions = []
    for pattern, extra, stmts in GLOSSARY:
        if re.search(pattern, core):
            expansions.append(extra)
            statements |= stmts
    return " ".join([core] + expansions), statements
