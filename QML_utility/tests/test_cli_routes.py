"""Installed interfaces must expose the same catalogs and project parsers."""
import json
import pytest
from qml_tricks.__main__ import main as research
from QML_utility.__main__ import main as utility


def test_catalog_parity_in_both_directions(capsys):
    research(['list'])
    expected=json.loads(capsys.readouterr().out)
    utility(['research','list'])
    assert json.loads(capsys.readouterr().out)==expected
    research(['project','strategies','--all'])
    assert json.loads(capsys.readouterr().out)==expected


@pytest.mark.parametrize('command',['setup','quickstart','run','evaluate','audit','predict','strategy'])
def test_project_help_reaches_same_parser(command,capsys):
    with pytest.raises(SystemExit) as direct:
        utility([command,'--help'])
    expected=capsys.readouterr().out
    with pytest.raises(SystemExit) as routed:
        research(['project',command,'--help'])
    assert direct.value.code==routed.value.code==0
    assert capsys.readouterr().out==expected


def test_invalid_project_command_has_usage_error():
    with pytest.raises(SystemExit) as error:
        research(['project','not-a-command'])
    assert error.value.code==2
