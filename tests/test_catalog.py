import pytest

from ai_share_of_voice.ui.catalog import (
    COUNTRIES,
    DEFAULT_CITY,
    DEFAULT_COUNTRY,
    DEFAULT_REGION,
    DEFAULT_SECTOR,
    OTHER_SECTOR,
    REGIONS,
    SECTORS,
    city_choices,
    country_choices,
    country_name,
    region_choices,
    sector_choices,
    sector_text,
)


def test_the_form_offers_spain_portugal_and_france():
    assert [code for _, code in country_choices("es")] == ["ES", "PT", "FR"]
    assert [label for label, _ in country_choices("en")] == ["Spain", "Portugal", "France"]


def test_every_country_has_regions_and_every_region_has_cities():
    assert set(REGIONS) == set(COUNTRIES)
    for country, regions in REGIONS.items():
        assert regions, country
        for region, cities in regions.items():
            assert cities, region
            assert len(cities) == len(set(cities)), region


def test_spain_lists_its_seventeen_autonomous_communities():
    assert len(REGIONS["ES"]) == 17


def test_cities_are_filtered_by_country_and_region():
    assert "Comunidad de Madrid" in region_choices("ES")
    assert "Madrid" in city_choices("ES", "Comunidad de Madrid")
    assert "Barcelona" not in city_choices("ES", "Comunidad de Madrid")
    assert "Porto" in city_choices("PT", "Norte")


def test_an_unknown_or_empty_choice_gives_no_options():
    assert region_choices(None) == []
    assert city_choices("ES", None) == []
    assert city_choices("XX", "Norte") == []


def test_the_defaults_are_consistent():
    assert DEFAULT_REGION in REGIONS[DEFAULT_COUNTRY]
    assert DEFAULT_CITY in REGIONS[DEFAULT_COUNTRY][DEFAULT_REGION]
    assert DEFAULT_SECTOR in SECTORS


def test_twelve_sectors_plus_other_in_both_languages():
    assert len(SECTORS) == 13 and list(SECTORS)[-1] == OTHER_SECTOR
    for labels in SECTORS.values():
        assert set(labels) == {"es", "en"}
    assert [key for _, key in sector_choices("en")] == list(SECTORS)


@pytest.mark.parametrize(
    ("key", "custom", "lang", "expected"),
    [
        ("food_delivery", "", "es", "Comida a domicilio"),
        ("food_delivery", "ignored", "en", "Food delivery"),
        (OTHER_SECTOR, "  peluquerías caninas ", "es", "peluquerías caninas"),
        (OTHER_SECTOR, "", "es", ""),  # the form validation then asks for a sector
    ],
)
def test_the_brief_receives_the_sector_as_text(key, custom, lang, expected):
    assert sector_text(key, custom, lang) == expected


def test_the_country_name_follows_the_language_of_the_questions():
    assert country_name("ES", "es") == "España"
    assert country_name("ES", "en") == "Spain"
    assert country_name("Andorra", "es") == "Andorra"
