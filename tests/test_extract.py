from app.config import Brand
from app.extract import (
    domain_of,
    extract_citations,
    extract_mentions,
    normalize_url,
)

OWN = Brand(name="Mochi HRMS")
COMPS = [Brand(name="Talenox"), Brand(name="Kakitangan"), Brand(name="Info-Tech"),
         Brand(name="PayrollPanda", aliases=["Payroll Panda"])]


def by_brand(text):
    return {m.brand: m for m in extract_mentions(text, OWN, COMPS)}


def test_hrms_alone_is_not_a_mention():
    assert by_brand("Any good HRMS will do. Mochi is a dessert.") == {}


def test_full_name_case_and_whitespace_insensitive():
    m = by_brand("We like **mochi   HRMS** a lot. Mochi HRMS again.")
    assert m["Mochi HRMS"].count == 2 and m["Mochi HRMS"].is_own


def test_word_boundaries_and_hyphens():
    assert "Talenox" not in by_brand("TalenoxPro is different")
    assert "Info-Tech" in by_brand("Try Info-Tech.")
    assert "Info-Tech" not in by_brand("Try Info-Techno")
    assert "Kakitangan" in by_brand("Kakitangan.com is popular")


def test_alias_counts_for_brand():
    assert by_brand("Payroll Panda is cheap")["PayrollPanda"].count == 1


def test_rank_first_is_order_of_first_mention():
    m = by_brand("Talenox and Mochi HRMS. Later Kakitangan, then Talenox again.")
    assert (m["Talenox"].rank_first, m["Mochi HRMS"].rank_first, m["Kakitangan"].rank_first) \
        == (1, 2, 3)


def test_rank_list_numbered_with_nested_items():
    text = (
        "Intro mentions Kakitangan first.\n\n"
        "1. **Talenox** - good\n"
        "   - compared with Kakitangan it is pricier\n"
        "2. **Mochi HRMS** - mobile app\n"
        "3. Kakitangan\n"
    )
    m = by_brand(text)
    assert m["Talenox"].rank_list == 1
    assert m["Mochi HRMS"].rank_list == 2
    assert m["Kakitangan"].rank_list == 1  # the nested line belongs to item 1
    assert m["Kakitangan"].rank_first == 1  # but it was named first in the intro


def test_rank_list_star_bullets():
    m = by_brand("* Talenox\n* Mochi HRMS\n")
    assert m["Mochi HRMS"].rank_list == 2


def test_rank_list_headings_win_over_bullets():
    text = "### Talenox\n- Pricing: cheap\n- Mobile: yes\n### Mochi HRMS\n- Pricing: fair\n"
    assert by_brand(text)["Mochi HRMS"].rank_list == 2


def test_rank_list_table_rows_skip_header():
    text = "| Vendor | Price |\n|---|---|\n| Talenox | $ |\n| Mochi HRMS | $$ |\n"
    m = by_brand(text)
    assert (m["Talenox"].rank_list, m["Mochi HRMS"].rank_list) == (1, 2)


def test_no_list_means_no_rank_list():
    assert by_brand("Mochi HRMS is fine.")["Mochi HRMS"].rank_list is None


def test_snippet_is_the_sentence():
    m = by_brand("First sentence. Then Mochi HRMS appears here! Final one.")
    assert m["Mochi HRMS"].snippet == "Then Mochi HRMS appears here!"


def test_normalize_url():
    assert normalize_url("https://WWW.G2.com/categories/hr/?utm_source=openai&x=1#top") \
        == "https://g2.com/categories/hr?x=1"
    assert normalize_url("http://talenox.com/") == "https://talenox.com"
    assert normalize_url("ftp://x.com") is None
    assert normalize_url("not a url") is None


def test_domain_of_uses_registered_domain():
    assert domain_of("https://blog.info-tech.com.my/a") == "info-tech.com.my"
    assert domain_of("https://g2.com/x") == "g2.com"


def test_citations_dedupe_per_kind_and_inline():
    cits = extract_citations(
        "See https://talenox.com/pricing. and [G2](https://www.g2.com/hr)",
        cited=["https://talenox.com/", "https://www.talenox.com"],
        retrieved=["https://talenox.com", "https://g2.com/hr?utm_medium=x"],
    )
    got = {(c.kind, c.url) for c in cits}
    assert got == {
        ("cited", "https://talenox.com"),
        ("retrieved", "https://talenox.com"),
        ("retrieved", "https://g2.com/hr"),
        ("inline", "https://talenox.com/pricing"),
        ("inline", "https://g2.com/hr"),
    }


def test_case_sensitive_brand_ignores_the_ordinary_word():
    kaki = Brand(name="Kakitangan", case_sensitive=True)
    text = "Pengurusan kakitangan is easier with Kakitangan.com. KAKITANGAN too."
    found = {m.brand: m for m in extract_mentions(text, kaki, [])}
    assert found["Kakitangan"].count == 1  # only the capitalised brand name, incl. "Kakitangan.com"
    assert extract_mentions("Urus kakitangan anda.", kaki, []) == []
