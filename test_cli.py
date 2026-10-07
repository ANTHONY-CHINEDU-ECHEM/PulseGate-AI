from pulsegate import cli


def test_no_arguments_prints_usage(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for name in ("webcam", "train", "evaluate", "serve", "analyze", "build_dataset"):
        assert name in out


def test_unknown_command_fails_cleanly(capsys):
    assert cli.main(["launch_rocket"]) == 2
    assert "unknown command" in capsys.readouterr().out


def test_malformed_option_fails_cleanly(capsys):
    assert cli.main(["info", "oops"]) == 2
    assert "key=value" in capsys.readouterr().out


def test_dotted_keys_are_configuration_and_plain_keys_are_arguments():
    overrides, args = cli._split_args(["train.epochs=2", "video=clip.mp4", "seconds=5"])
    assert overrides == {"train.epochs": 2}
    assert args == {"video": "clip.mp4", "seconds": 5}


def test_info_runs(capsys):
    assert cli.main(["info"]) == 0
    assert "PulseGate AI" in capsys.readouterr().out


def test_every_command_has_a_summary():
    assert all(summary and callable(func) for func, summary in cli.COMMANDS.values())
