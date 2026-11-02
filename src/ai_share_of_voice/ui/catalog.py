"""Choices of the Brief form: countries, regions, main cities and sectors.

Values are stable keys (``ES``, ``food_delivery``); labels change with the language, so
switching the interface language keeps what the user picked. Region and city names are
proper nouns and stay in their own language. Cities are the best known of each region,
not a full list: the form stays short and the AI knows those places.

The brief receives plain text in the language of the questions: "comida a domicilio" for a
Spanish panel, "food delivery" for an English one, and the country name for web searches.
"""

from ai_share_of_voice.ui.i18n import DEFAULT_LANG

OTHER_SECTOR = "other"

COUNTRIES: dict[str, dict[str, str]] = {
    "ES": {"es": "España", "en": "Spain"},
    "PT": {"es": "Portugal", "en": "Portugal"},
    "FR": {"es": "Francia", "en": "France"},
}

REGIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "ES": {
        "Andalucía": ("Sevilla", "Málaga", "Granada", "Córdoba", "Cádiz", "Almería", "Marbella"),
        "Aragón": ("Zaragoza", "Huesca", "Teruel"),
        "Principado de Asturias": ("Oviedo", "Gijón", "Avilés"),
        "Islas Baleares": ("Palma", "Ibiza", "Mahón"),
        "Canarias": ("Las Palmas de Gran Canaria", "Santa Cruz de Tenerife", "La Laguna"),
        "Cantabria": ("Santander", "Torrelavega"),
        "Castilla-La Mancha": ("Toledo", "Albacete", "Ciudad Real", "Guadalajara", "Cuenca"),
        "Castilla y León": ("Valladolid", "Burgos", "Salamanca", "León", "Segovia"),
        "Cataluña": ("Barcelona", "Tarragona", "Girona", "Lleida", "Sabadell", "Terrassa"),
        "Comunidad de Madrid": (
            "Madrid",
            "Alcalá de Henares",
            "Móstoles",
            "Getafe",
            "Leganés",
            "Alcobendas",
        ),
        "Comunidad Valenciana": (
            "Valencia",
            "Alicante",
            "Elche",
            "Castellón de la Plana",
            "Benidorm",
        ),
        "Extremadura": ("Badajoz", "Cáceres", "Mérida"),
        "Galicia": ("Vigo", "A Coruña", "Santiago de Compostela", "Ourense", "Lugo"),
        "Región de Murcia": ("Murcia", "Cartagena", "Lorca"),
        "Navarra": ("Pamplona", "Tudela"),
        "País Vasco": ("Bilbao", "Vitoria-Gasteiz", "San Sebastián"),
        "La Rioja": ("Logroño",),
    },
    "PT": {
        "Norte": ("Porto", "Braga", "Guimarães", "Vila Nova de Gaia", "Viana do Castelo"),
        "Centro": ("Coimbra", "Aveiro", "Viseu", "Leiria"),
        "Área Metropolitana de Lisboa": ("Lisboa", "Sintra", "Cascais", "Almada", "Setúbal"),
        "Alentejo": ("Évora", "Beja"),
        "Algarve": ("Faro", "Albufeira", "Portimão", "Lagos"),
        "Açores": ("Ponta Delgada",),
        "Madeira": ("Funchal",),
    },
    "FR": {
        "Île-de-France": ("Paris", "Versailles", "Boulogne-Billancourt"),
        "Auvergne-Rhône-Alpes": ("Lyon", "Grenoble", "Saint-Étienne", "Clermont-Ferrand", "Annecy"),
        "Provence-Alpes-Côte d'Azur": ("Marseille", "Nice", "Toulon", "Aix-en-Provence", "Cannes"),
        "Occitanie": ("Toulouse", "Montpellier", "Nîmes", "Perpignan"),
        "Nouvelle-Aquitaine": ("Bordeaux", "Limoges", "Poitiers", "La Rochelle", "Biarritz"),
        "Hauts-de-France": ("Lille", "Amiens", "Roubaix"),
        "Grand Est": ("Strasbourg", "Reims", "Metz", "Nancy"),
        "Pays de la Loire": ("Nantes", "Angers", "Le Mans"),
        "Bretagne": ("Rennes", "Brest", "Quimper"),
        "Normandie": ("Rouen", "Caen", "Le Havre"),
        "Bourgogne-Franche-Comté": ("Dijon", "Besançon"),
        "Centre-Val de Loire": ("Tours", "Orléans"),
        "Corse": ("Ajaccio", "Bastia"),
    },
}

SECTORS: dict[str, dict[str, str]] = {
    "food_delivery": {"es": "Comida a domicilio", "en": "Food delivery"},
    "restaurants": {"es": "Restaurantes y bares", "en": "Restaurants and bars"},
    "fashion": {"es": "Moda y tiendas de ropa", "en": "Fashion and clothing stores"},
    "groceries": {"es": "Supermercados y alimentación", "en": "Supermarkets and groceries"},
    "banking": {"es": "Banca y seguros", "en": "Banking and insurance"},
    "telecom": {"es": "Telefonía e internet", "en": "Mobile and internet providers"},
    "travel": {"es": "Viajes y hoteles", "en": "Travel and hotels"},
    "health": {"es": "Salud y clínicas", "en": "Health and clinics"},
    "education": {"es": "Educación y formación", "en": "Education and training"},
    "electronics": {"es": "Tecnología y electrónica", "en": "Technology and electronics"},
    "home": {"es": "Hogar y reformas", "en": "Home and renovation"},
    "mobility": {"es": "Automoción y movilidad", "en": "Cars and mobility"},
    OTHER_SECTOR: {"es": "Otro (escríbelo)", "en": "Other (type it)"},
}

DEFAULT_COUNTRY = "ES"
DEFAULT_REGION = "Comunidad de Madrid"
DEFAULT_CITY = "Madrid"
DEFAULT_SECTOR = "food_delivery"


def _label(labels: dict[str, str], lang: str) -> str:
    return labels.get(lang, labels[DEFAULT_LANG])


def country_choices(lang: str) -> list[tuple[str, str]]:
    return [(_label(names, lang), code) for code, names in COUNTRIES.items()]


def region_choices(country: str | None) -> list[str]:
    return list(REGIONS.get(country or "", {}))


def city_choices(country: str | None, region: str | None) -> list[str]:
    return list(REGIONS.get(country or "", {}).get(region or "", ()))


def sector_choices(lang: str) -> list[tuple[str, str]]:
    return [(_label(names, lang), key) for key, names in SECTORS.items()]


def country_name(code: str, lang: str) -> str:
    """The country as written in ``lang``; an unknown code is returned as it is."""
    return _label(COUNTRIES[code], lang) if code in COUNTRIES else code


def sector_text(key: str, custom: str, lang: str) -> str:
    """The sector the brief receives: the typed text for "other", else the label in ``lang``."""
    if key == OTHER_SECTOR or key not in SECTORS:
        return custom.strip()
    return _label(SECTORS[key], lang)
