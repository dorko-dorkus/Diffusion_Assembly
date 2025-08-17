import json

from loto import logging_setup


def test_json_log_contains_context_and_message(capsys):
    logging_setup.init_logging(verbosity=1)
    logger = logging_setup.get_logger().bind(
        wo="w0", asset="assetA", rule_hash="hash123"
    )
    logger.info("hello")
    captured = capsys.readouterr()
    line = captured.out.strip()
    data = json.loads(line)
    assert data["wo"] == "w0"
    assert data["asset"] == "assetA"
    assert data["rule_hash"] == "hash123"
    assert data["message"] == "hello"
    assert data["level"] == "INFO"


def test_verbosity_respected(capsys):
    logging_setup.init_logging(verbosity=0)
    logger = logging_setup.get_logger()
    logger.info("should be hidden")
    assert capsys.readouterr().out == ""

    logging_setup.init_logging(verbosity=2)
    logger = logging_setup.get_logger()
    logger.debug("now visible")
    line = capsys.readouterr().out.strip()
    data = json.loads(line)
    assert data["level"] == "DEBUG"
    assert data["message"] == "now visible"
