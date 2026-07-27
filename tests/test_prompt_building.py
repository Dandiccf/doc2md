"""How the vision prompt is assembled from the config.

Unit-level: no network, no API key, no sample downloads — just the string the
model would be sent.

The point of contention is the relationship between two options.
``structured_description`` fixes the *shape* of the answer (a JSON object with
``summary`` and ``detail``); ``picture_description_prompt`` says what the model
should look for. They are independent, but the structured branch used to return
a fully hardcoded prompt, so enabling it silently discarded the caller's
instruction — no error, no warning, nothing in the config docs. You could set a
prompt, watch it work in plain mode, turn on structured mode, and watch it stop
mattering.
"""

from __future__ import annotations

import pytest

from doc2md.config import (
    DEFAULT_PICTURE_DESCRIPTION_PROMPT,
    DEFAULT_STRUCTURED_DETAIL_PROMPT,
    PipelineConfig,
)
from doc2md.converter import DocumentPipeline

# A prompt for the case the default gets wrong: pictures that ARE the subject,
# where the objects have to be named so they can be searched for.
EIGENER_PROMPT = (
    "List every object visible in this image individually, with counts and any "
    "labels or text on them. Do not interpret; name what is there."
)


def _prompt(**config_kwargs) -> str:
    return DocumentPipeline(PipelineConfig(**config_kwargs))._build_base_prompt()


class TestPlainMode:
    def test_the_configured_prompt_is_the_prompt(self) -> None:
        assert _prompt(picture_description_prompt=EIGENER_PROMPT) == EIGENER_PROMPT

    def test_the_default_is_used_when_nothing_is_configured(self) -> None:
        assert _prompt() == DEFAULT_PICTURE_DESCRIPTION_PROMPT


class TestStructuredMode:
    def test_an_untouched_config_yields_the_prompt_it_always_did(self) -> None:
        """Backward compatibility, and the reason the two defaults are separate.

        Substituting the plain-mode default into ``detail`` would have reworded
        the prompt for everyone already using structured descriptions. It stays
        verbatim unless the caller asks for something else.
        """
        prompt = _prompt(structured_description=True)

        assert prompt.endswith(DEFAULT_STRUCTURED_DETAIL_PROMPT)
        assert DEFAULT_PICTURE_DESCRIPTION_PROMPT not in prompt

    def test_a_custom_prompt_now_reaches_the_model(self) -> None:
        """The regression this file exists for."""
        prompt = _prompt(
            structured_description=True, picture_description_prompt=EIGENER_PROMPT
        )

        assert EIGENER_PROMPT in prompt
        assert DEFAULT_STRUCTURED_DETAIL_PROMPT not in prompt

    def test_the_json_contract_survives_a_custom_prompt(self) -> None:
        """The shape of the answer is not the caller's to change — the
        serializer parses these two fields."""
        prompt = _prompt(
            structured_description=True, picture_description_prompt=EIGENER_PROMPT
        )

        assert "JSON object" in prompt
        assert '"summary"' in prompt
        assert '"detail"' in prompt

    def test_the_custom_instruction_drives_detail_not_summary(self) -> None:
        vor_detail, _, nach_detail = _prompt(
            structured_description=True, picture_description_prompt=EIGENER_PROMPT
        ).partition('- "detail":')

        assert EIGENER_PROMPT not in vor_detail
        assert EIGENER_PROMPT in nach_detail

    @pytest.mark.parametrize("leer", ["", "   "])
    def test_an_empty_prompt_is_taken_literally(self, leer: str) -> None:
        """Not the same as "unset": an explicit empty string differs from the
        default, so it is honoured rather than replaced. Documenting the edge
        rather than defending against it — the caller asked for nothing."""
        prompt = _prompt(structured_description=True, picture_description_prompt=leer)

        assert DEFAULT_STRUCTURED_DETAIL_PROMPT not in prompt
        assert prompt.rstrip().endswith('- "detail":')


class TestContextualPrompt:
    """The base prompt is wrapped with document context and a language hint;
    the custom instruction has to survive that too."""

    def test_the_custom_instruction_survives_context_and_language(self) -> None:
        pipeline = DocumentPipeline(
            PipelineConfig(
                structured_description=True,
                picture_description_prompt=EIGENER_PROMPT,
                picture_description_lang="de",
            )
        )

        prompt = pipeline._build_contextual_prompt(
            doc_title="Lichtbildbeilage", surrounding_text="Sichergestellte Gegenstände"
        )

        assert EIGENER_PROMPT in prompt
        assert "Lichtbildbeilage" in prompt
        assert "Sichergestellte Gegenstände" in prompt
