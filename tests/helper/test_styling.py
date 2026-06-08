"""Tests for shared console styling helpers."""

from __future__ import annotations

from io import StringIO

from rich.text import Text

from biomodals.helper.styling import (
    console_color_enabled,
    print_rich,
    strip_ansi,
    styled_text,
)


def test_console_color_enabled_honors_no_color() -> None:
    """Color settings should honor NO_COLOR and the requested env var."""
    assert console_color_enabled({"NO_COLOR": "1"}) is False
    assert (
        console_color_enabled(
            {"BIOMODALS_WORKFLOW_COLOR": "false"},
            color_env_var="BIOMODALS_WORKFLOW_COLOR",
        )
        is False
    )
    assert console_color_enabled({}) is True


def test_print_rich_uses_auto_detection_by_default_for_captured_output() -> None:
    """Default Rich output should be plain when the output stream is captured."""
    stream = StringIO()

    print_rich(
        "workflow ok",
        style="green",
        file=stream,
        environ={},
        color_env_var="BIOMODALS_WORKFLOW_COLOR",
    )

    assert stream.getvalue() == "workflow ok\n"


def test_print_rich_emits_ansi_when_color_is_forced() -> None:
    """Forced-color Rich output should include ANSI escapes."""
    stream = StringIO()

    print_rich(
        "workflow ok",
        style="green",
        file=stream,
        environ={"BIOMODALS_WORKFLOW_COLOR": "1"},
        color_env_var="BIOMODALS_WORKFLOW_COLOR",
    )

    output = stream.getvalue()
    assert "\x1b[" in output
    assert strip_ansi(output) == "workflow ok\n"


def test_print_rich_suppresses_ansi_when_no_color_is_set() -> None:
    """NO_COLOR should suppress ANSI escapes for string renderables."""
    stream = StringIO()

    print_rich(
        "workflow ok",
        style="green",
        file=stream,
        environ={"NO_COLOR": "1"},
        color_env_var="BIOMODALS_WORKFLOW_COLOR",
    )

    assert stream.getvalue() == "workflow ok\n"


def test_print_rich_suppresses_text_segment_styles_when_no_color_is_set() -> None:
    """NO_COLOR should suppress ANSI escapes for styled text renderables."""
    stream = StringIO()

    print_rich(
        styled_text(("node-a", "bold yellow"), (" <- node-b", "grey50")),
        file=stream,
        environ={"NO_COLOR": "1"},
        color_env_var="BIOMODALS_WORKFLOW_COLOR",
    )

    assert stream.getvalue() == "node-a <- node-b\n"


def test_styled_text_preserves_plain_text_and_segment_styles() -> None:
    """Segmented text should preserve both plain text and requested styles."""
    text = styled_text(
        ("[workflow]   ", "grey50"),
        ("node-a", "bold yellow"),
        (" [", "grey50"),
        ("remote", "bold"),
        ("; ", "grey50"),
        ("DemoNode", "bold"),
        ("] <- ", "grey50"),
        ("node-b", "grey50"),
    )

    assert isinstance(text, Text)
    assert text.plain == "[workflow]   node-a [remote; DemoNode] <- node-b"
    assert [span.style for span in text.spans] == [
        "grey50",
        "bold yellow",
        "grey50",
        "bold",
        "grey50",
        "bold",
        "grey50",
        "grey50",
    ]
