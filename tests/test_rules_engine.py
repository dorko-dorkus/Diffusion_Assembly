import pytest

from loto import rule_engine

VALID = """
version: 1.2.3
domains:
  steam:
    min_isolation: double_block_and_bleed
    verify:
      - lock valve
      - depressurize
  condensate:
    min_isolation: single_block
    verify:
      - close valve
  instrument_air:
    min_isolation: single_block
    verify:
      - close valve
"""


def test_parse_valid_rules():
    pack = rule_engine.load(VALID)
    assert pack.version == "1.2.3"
    assert set(pack.domains.keys()) == {"steam", "condensate", "instrument_air"}
    assert pack.domains["steam"].verify[0] == "lock valve"


def test_hash_stable_under_key_permutation():
    permuted = """
domains:
  instrument_air:
    verify:
      - close valve
    min_isolation: single_block
  condensate:
    verify:
      - close valve
    min_isolation: single_block
  steam:
    min_isolation: double_block_and_bleed
    verify:
      - lock valve
      - depressurize
version: 1.2.3
"""
    pack1 = rule_engine.load(VALID)
    pack2 = rule_engine.load(permuted)
    assert pack1.hash() == pack2.hash()


@pytest.mark.parametrize(
    "bad_yaml, path",
    [
        # invalid version format
        (
            """
version: 1
domains:
  steam:
    min_isolation: double_block_and_bleed
    verify:
      - step
  condensate:
    min_isolation: single_block
    verify:
      - step
  instrument_air:
    min_isolation: single_block
    verify:
      - step
""",
            "version",
        ),
        # missing required domain
        (
            """
version: 1.2.3
domains:
  steam:
    min_isolation: double_block_and_bleed
    verify:
      - step
  condensate:
    min_isolation: single_block
    verify:
      - step
""",
            "domains",
        ),
        # bad min_isolation value
        (
            """
version: 1.2.3
domains:
  steam:
    min_isolation: bogus
    verify:
      - step
  condensate:
    min_isolation: single_block
    verify:
      - step
  instrument_air:
    min_isolation: single_block
    verify:
      - step
""",
            "domains.steam.min_isolation",
        ),
        # missing verify steps
        (
            """
version: 1.2.3
domains:
  steam:
    min_isolation: double_block_and_bleed
    verify:
      - step
  condensate:
    min_isolation: single_block
  instrument_air:
    min_isolation: single_block
    verify:
      - step
""",
            "domains.condensate.verify",
        ),
        # extra top-level key
        (
            """
version: 1.2.3
extra: 1
domains:
  steam:
    min_isolation: double_block_and_bleed
    verify:
      - step
  condensate:
    min_isolation: single_block
    verify:
      - step
  instrument_air:
    min_isolation: single_block
    verify:
      - step
""",
            "extra",
        ),
    ],
)
def test_invalid_rules(bad_yaml, path):
    with pytest.raises(rule_engine.RulesError) as exc:
        rule_engine.load(bad_yaml)
    assert exc.value.code == "RULES/INVALID"
    assert path in exc.value.hint
