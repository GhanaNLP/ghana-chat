"""Ghanaian Abbreviations and Acronyms Resolution Engine for Ghana Chat.

Detects acronyms/abbreviations in user queries, resolves them to full official names,
and maps full names back to common abbreviations.

Expansions are:
1. Used by the retriever to search the Knowledge Graph under BOTH full name and abbreviation.
2. Supplied as context to the model so it understands the exact meaning.
"""

import re
from typing import Dict, List, Tuple, Set

# Comprehensive dictionary of Ghanaian institutions, political parties, ministries, and agencies
GHANA_ABBREVIATIONS: Dict[str, str] = {
    # Governance & Planning
    "NDPC": "National Development Planning Commission",
    "EC": "Electoral Commission",
    "NCCE": "National Commission for Civic Education",
    "CHRAJ": "Commission on Human Rights and Administrative Justice",
    "PAC": "Public Accounts Committee",
    "NIA": "National Identification Authority",
    "NITA": "National Information Technology Agency",
    "BOG": "Bank of Ghana",
    "BOGH": "Bank of Ghana",

    # Political Parties
    "NDC": "National Democratic Congress",
    "NPP": "New Patriotic Party",
    "CPP": "Convention People's Party",
    "PNC": "People's National Convention",
    "PPP": "Progressive People's Party",
    "GCPP": "Great Consolidated Popular Party",

    # Revenue, Finance & Social Security
    "GRA": "Ghana Revenue Authority",
    "SSNIT": "Social Security and National Insurance Trust",
    "COCOBOD": "Ghana Cocoa Board",
    "GIPC": "Ghana Investment Promotion Centre",
    "SEC": "Securities and Exchange Commission",
    "PPA": "Public Procurement Authority",

    # Utilities, Energy & Infrastructure
    "ECG": "Electricity Company of Ghana",
    "VRA": "Volta River Authority",
    "GRIDCO": "Ghana Grid Company",
    "BPA": "Bui Power Authority",
    "GNPC": "Ghana National Petroleum Corporation",
    "BOST": "Bulk Oil Storage and Transportation",
    "NPA": "National Petroleum Authority",
    "GWCL": "Ghana Water Company Limited",
    "PURC": "Public Utilities Regulatory Commission",
    "PDS": "Power Distribution Services",
    "DVLA": "Driver and Vehicle Licensing Authority",
    "NRSA": "National Road Safety Authority",
    "GPHA": "Ghana Ports and Harbours Authority",
    "GACL": "Ghana Airports Company Limited",

    # Education & Research
    "GES": "Ghana Education Service",
    "GTEC": "Ghana Tertiary Education Commission",
    "WAEC": "West African Examinations Council",
    "NSS": "National Service Scheme",
    "SLTF": "Students Loan Trust Fund",
    "CSIR": "Council for Scientific and Industrial Research",
    "KNUST": "Kwame Nkrumah University of Science and Technology",
    "UG": "University of Ghana",
    "UCC": "University of Cape Coast",
    "UDS": "University for Development Studies",
    "UPSA": "University of Professional Studies, Accra",
    "UEW": "University of Education, Winneba",
    "UMaT": "University of Mines and Technology",
    "UHAS": "University of Health and Allied Sciences",
    "UENR": "University of Energy and Natural Resources",

    # Health & Safety
    "GHS": "Ghana Health Service",
    "FDA": "Food and Drugs Authority",
    "NHIA": "National Health Insurance Authority",
    "NHIS": "National Health Insurance Scheme",
    "EPA": "Environmental Protection Agency",
    "NADMO": "National Disaster Management Organisation",
    "GNFS": "Ghana National Fire Service",
    "GPS": "Ghana Police Service",
    "GAF": "Ghana Armed Forces",
    "GIS": "Ghana Immigration Service",

    # Media & Associations
    "GJA": "Ghana Journalists Association",
    "GIBA": "Ghana Independent Broadcasters Association",
    "PRINPAG": "Private Newspaper and Online News Publishers Association of Ghana",
    "GCNH": "Ghana Chamber of Construction Industry",
    "AGI": "Association of Ghana Industries",
    "TUC": "Trades Union Congress",
    "GNAT": "Ghana National Association of Teachers",
    "NAGRAT": "National Association of Graduate Teachers",
    "CCT": "Coalition of Concerned Teachers",
    "GMA": "Ghana Medical Association",
    "GRNMA": "Ghana Registered Nurses and Midwives Association",

    # Ministries
    "MOE": "Ministry of Education",
    "MOF": "Ministry of Finance",
    "MOFEP": "Ministry of Finance and Economic Planning",
    "MOH": "Ministry of Health",
    "MLNR": "Ministry of Lands and Natural Resources",
    "MOTI": "Ministry of Trade and Industry",
    "MOFA": "Ministry of Food and Agriculture",
    "MOC": "Ministry of Communications and Digitalisation",
    "MINT": "Ministry of the Interior",
    "MOD": "Ministry of Defence",
    "MOJ": "Ministry of Justice and Attorney-General's Department",
    "MOTAC": "Ministry of Tourism, Arts and Culture",
    "MOYS": "Ministry of Youth and Sports",
    "MLGDRD": "Ministry of Local Government, Decentralisation and Rural Development",

    # Regional & International Organizations Active in Ghana
    "GIZ": "Deutsche Gesellschaft für Internationale Zusammenarbeit",
    "UNHCR": "United Nations High Commissioner for Refugees",
    "UNICEF": "United Nations Children's Fund",
    "UNDP": "United Nations Development Programme",
    "WHO": "World Health Organization",
    "JICA": "Japan International Cooperation Agency",
    "USAID": "United States Agency for International Development",
    "AFD": "Agence Française de Développement",
    "ECOWAS": "Economic Community of West African States",
    "AU": "African Union",
    "AfCFTA": "African Continental Free Trade Area",
}

# Reverse mapping: Full name (lowercase) -> Acronym
FULL_NAME_TO_ABBREV: Dict[str, str] = {
    full.lower(): abbr for abbr, full in GHANA_ABBREVIATIONS.items()
}


class AbbreviationResolver:
    """Resolves abbreviations to full names and vice versa."""

    def __init__(self, custom_mapping: Dict[str, str] = None):
        self.abbrev_map = dict(GHANA_ABBREVIATIONS)
        if custom_mapping:
            self.abbrev_map.update(custom_mapping)
        self.reverse_map = {v.lower(): k for k, v in self.abbrev_map.items()}

    def resolve_query(self, query: str) -> Tuple[List[Tuple[str, str]], Set[str]]:
        """Identify abbreviations in query and return:

        1. List of (abbreviation, full_name) pairs found.
        2. Set of entity search aliases (abbreviation + full name) for retrieval.
        """
        found_pairs: List[Tuple[str, str]] = []
        search_aliases: Set[str] = set()

        # Check for exact acronym tokens in query
        words = re.findall(r"\b[A-Za-z0-9&'-]+\b", query)
        for idx, t in enumerate(words):
            t_upper = t.upper()
            # If acronym is a common English word/stopword (like WHO, IT, AS),
            # only treat as acronym if it is ALL-CAPS in the query and not merely sentence-capitalized
            if t.lower() in {"who", "it", "as", "at", "in", "an", "am", "be", "do", "no", "so", "to", "us"}:
                if not (t.isupper() and (idx > 0 or len(words) == 1)):
                    continue

            if t_upper in self.abbrev_map:
                full = self.abbrev_map[t_upper]
                pair = (t_upper, full)
                if pair not in found_pairs:
                    found_pairs.append(pair)
                # Keep both acronym and full name
                search_aliases.add(t_upper.lower())
                search_aliases.add(full.lower())

        # Check if full name is mentioned in query
        q_lower = query.lower()
        for full_lower, abbr in self.reverse_map.items():
            if full_lower in q_lower:
                pair = (abbr, self.abbrev_map[abbr])
                if pair not in found_pairs:
                    found_pairs.append(pair)
                search_aliases.add(abbr.lower())
                search_aliases.add(full_lower)

        return found_pairs, search_aliases

    def format_abbreviations_context(self, pairs: List[Tuple[str, str]]) -> str:
        """Format identified abbreviations into prompt context for the LLM."""
        if not pairs:
            return ""
        lines = [f"- {abbr}: {full}" for abbr, full in pairs]
        return "\n".join(lines)
