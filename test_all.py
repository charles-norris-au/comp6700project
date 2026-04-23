"""
test_all.py
-----------
Pytest test suite for:
  - kde_extractor.py  (Task 1)
  - kde_compare.py    (Task 2)
  - task3.py          (Task 3)

LLM-touching functions (run_llm, extract_kdes_from_pdf, process_two_files)
are tested with a mocked pipeline so no model is loaded and tests finish
in seconds.  PDF tests use tiny single-sentence PDFs built with fitz.
"""

# ---------------------------------------------------------------------------
# Stub out heavy dependencies (torch, transformers) so the test suite runs
# without a GPU environment or model downloads.
# ---------------------------------------------------------------------------
import sys
from unittest.mock import MagicMock

for _mod in ("torch", "transformers", "transformers.pipeline"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Give torch.bfloat16 a concrete sentinel value used in kde_extractor
import torch as _torch
_torch.bfloat16 = "bfloat16"

import csv
import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import fitz
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(__file__))

import task1 as ex
import task2 as cmp
import task3        as t3


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture()
def tmp(tmp_path):
    """Convenience alias – every test gets a fresh temp directory."""
    return tmp_path


def make_pdf(path: str, text: str) -> str:
    """Create a minimal single-page PDF containing *text* at *path*."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    doc.save(path)
    doc.close()
    return path


def mock_pipe_returning(yaml_text: str):
    """
    Return a callable that mimics the HuggingFace text-generation pipeline.
    The assistant reply is a plain string (the real Gemma shape).
    """
    def pipe(messages, max_new_tokens=1024):
        return [{
            "generated_text": [
                {"role": "system",  "content": "system prompt"},
                {"role": "user",    "content": "user prompt"},
                {"role": "assistant","content": yaml_text},
            ]
        }]
    return pipe


SIMPLE_YAML = (
    "element1:\n"
    "  name: Anonymous Auth\n"
    "  requirements:\n"
    "    - Disable anonymous access\n"
    "    - Enforce authentication\n"
)


# =============================================================================
# KDE_EXTRACTOR – Task 1
# =============================================================================

class TestExtractPdfTextFast:
    def test_returns_text_from_valid_pdf(self, tmp):
        pdf = make_pdf(str(tmp / "sample.pdf"), "Hello from PDF")
        result = ex.extract_pdf_text_fast(str(tmp / "sample.pdf"))
        assert "Hello from PDF" in result

    def test_returns_empty_string_for_missing_file(self):
        result = ex.extract_pdf_text_fast("/nonexistent/path/file.pdf")
        assert result == ""

    def test_returns_empty_string_for_corrupt_file(self, tmp):
        bad = tmp / "bad.pdf"
        bad.write_bytes(b"not a pdf at all")
        result = ex.extract_pdf_text_fast(str(bad))
        assert result == ""

    def test_multipage_pdf_joins_all_pages(self, tmp):
        doc = fitz.open()
        for word in ("Alpha", "Beta", "Gamma"):
            p = doc.new_page()
            p.insert_text((50, 50), word)
        path = str(tmp / "multi.pdf")
        doc.save(path)
        doc.close()
        result = ex.extract_pdf_text_fast(path)
        for word in ("Alpha", "Beta", "Gamma"):
            assert word in result


class TestChunkText:
    def test_single_chunk_when_text_fits(self):
        chunks = ex.chunk_text("line1\nline2\nline3", max_chars=100)
        assert len(chunks) == 1

    def test_splits_into_multiple_chunks(self):
        # 5 paragraphs of 10 chars each → max_chars=25 forces splits
        text = "\n".join(["1234567890"] * 5)
        chunks = ex.chunk_text(text, max_chars=25)
        assert len(chunks) > 1

    def test_no_empty_chunks_produced(self):
        text = "\n\n\nsome text\n\n\n"
        chunks = ex.chunk_text(text, max_chars=50)
        assert all(c.strip() for c in chunks)

    def test_oversized_paragraph_is_hard_split(self):
        big = "A" * 500
        chunks = ex.chunk_text(big, max_chars=100)
        assert all(len(c) <= 100 for c in chunks)

    def test_empty_text_returns_no_chunks(self):
        assert ex.chunk_text("") == []

    def test_exact_boundary_paragraph_stays_in_one_chunk(self):
        # paragraph is exactly max_chars; should produce one chunk
        text = "X" * 50
        chunks = ex.chunk_text(text, max_chars=50)
        assert len(chunks) == 1


class TestPromptBuilders:
    """Prompt builders are pure functions – verify structure and text inclusion."""

    SAMPLE = "The kubelet must disable anonymous auth."

    def test_zero_shot_contains_text(self):
        p = ex.build_kde_prompt_zero_shot(self.SAMPLE)
        assert self.SAMPLE in p
        assert "element1" in p
        assert "YAML" in p

    def test_one_shot_contains_example_and_text(self):
        p = ex.build_kde_prompt_one_shot(self.SAMPLE)
        assert self.SAMPLE in p
        assert "Example" in p
        assert "element1" in p

    def test_chain_of_thought_contains_text(self):
        p = ex.build_kde_prompt_chain_of_thought(self.SAMPLE)
        assert self.SAMPLE in p
        assert "step-by-step" in p
        assert "element1" in p

    def test_builders_return_strings(self):
        for builder in (
            ex.build_kde_prompt_zero_shot,
            ex.build_kde_prompt_one_shot,
            ex.build_kde_prompt_chain_of_thought,
        ):
            assert isinstance(builder(self.SAMPLE), str)


class TestRunLlm:
    """Mock the pipeline – no model loading."""

    def test_extracts_plain_string_assistant_content(self):
        pipe = mock_pipe_returning("my yaml output")
        result = ex.run_llm(pipe, "prompt")
        assert result == "my yaml output"

    def test_extracts_list_of_dicts_assistant_content(self):
        def pipe(messages, max_new_tokens=1024):
            return [{
                "generated_text": [
                    {"role": "system",   "content": []},
                    {"role": "user",     "content": []},
                    {"role": "assistant","content": [{"text": "yaml content"}]},
                ]
            }]
        result = ex.run_llm(pipe, "prompt")
        assert result == "yaml content"

    def test_returns_raw_string_for_non_chat_pipeline(self):
        def pipe(messages, max_new_tokens=1024):
            return [{"generated_text": "raw text output"}]
        result = ex.run_llm(pipe, "prompt")
        assert result == "raw text output"

    def test_logs_warn_for_unknown_format(self, capsys):
        def pipe(messages, max_new_tokens=1024):
            return [{"unexpected_key": "value"}]
        result = ex.run_llm(pipe, "prompt")
        assert "[WARN]" in capsys.readouterr().out

    def test_uses_last_turn_not_first(self):
        """Assistant reply must come from the LAST generated turn."""
        def pipe(messages, max_new_tokens=1024):
            return [{
                "generated_text": [
                    {"role": "user",      "content": "WRONG"},
                    {"role": "assistant", "content": "CORRECT"},
                ]
            }]
        assert ex.run_llm(pipe, "prompt") == "CORRECT"


class TestExtractYamlBlock:
    def test_fenced_yaml_block(self):
        text = "some prose\n```yaml\nelement1:\n  name: Foo\n```\nmore prose"
        result = ex.extract_yaml_block(text)
        assert result == "element1:\n  name: Foo"

    def test_generic_fenced_block(self):
        text = "```\nelement1:\n  name: Bar\n```"
        result = ex.extract_yaml_block(text)
        assert result == "element1:\n  name: Bar"

    def test_raw_yaml_fallback(self):
        text = "Here is the output:\nelement1:\n  name: Baz\n  requirements:\n    - req1\n"
        result = ex.extract_yaml_block(text)
        assert result is not None
        assert "element1" in result

    def test_returns_none_for_no_yaml(self):
        assert ex.extract_yaml_block("just plain prose with no yaml") is None

    def test_trailing_prose_not_captured(self):
        text = "element1:\n  name: X\n  requirements:\n    - r1\n\nThis is trailing prose."
        result = ex.extract_yaml_block(text)
        assert result is not None
        assert "trailing prose" not in result


class TestParseKdeYaml:
    def test_parses_valid_yaml(self):
        result = ex.parse_kde_yaml(SIMPLE_YAML)
        assert "element1" in result
        assert result["element1"]["name"] == "Anonymous Auth"
        assert "Disable anonymous access" in result["element1"]["requirements"]

    def test_filters_numeric_requirements_int(self):
        yaml_text = (
            "element1:\n"
            "  name: Audit Logging\n"
            "  requirements:\n"
            "    - Enable audit logs\n"
            "    - Retain logs\n"
            "    - 20\n"
            "    - 42\n"
        )
        result = ex.parse_kde_yaml(yaml_text)
        reqs = result.get("element1", {}).get("requirements", [])
        assert "20" not in reqs
        assert "42" not in reqs
        assert "Enable audit logs" in reqs

    def test_filters_numeric_requirements_float(self):
        yaml_text = (
            "element1:\n"
            "  name: TLS Config\n"
            "  requirements:\n"
            "    - Use TLS 1.2\n"
            "    - Rotate certs\n"
            "    - 3.2\n"
        )
        result = ex.parse_kde_yaml(yaml_text)
        reqs = result.get("element1", {}).get("requirements", [])
        assert "3.2" not in reqs

    def test_drops_element_with_fewer_than_two_real_reqs(self, capsys):
        yaml_text = (
            "element1:\n"
            "  name: Noisy Element\n"
            "  requirements:\n"
            "    - Manual\n"
            "    - 20\n"
            "    - 30\n"
        )
        result = ex.parse_kde_yaml(yaml_text)
        assert result == {}
        assert "[SKIP]" in capsys.readouterr().out

    def test_deduplicates_requirements(self):
        yaml_text = (
            "element1:\n"
            "  name: RBAC Roles\n"
            "  requirements:\n"
            "    - Restrict access\n"
            "    - Restrict access\n"
            "    - Avoid wildcards\n"
        )
        result = ex.parse_kde_yaml(yaml_text)
        reqs = result["element1"]["requirements"]
        assert reqs.count("Restrict access") == 1

    def test_normalises_elementX_to_element1(self):
        yaml_text = (
            "elementX:\n"
            "  name: Token Mount\n"
            "  requirements:\n"
            "    - Only mount where needed\n"
            "    - Review periodically\n"
        )
        result = ex.parse_kde_yaml(yaml_text)
        assert "element1" in result

    def test_returns_empty_for_non_yaml(self):
        assert ex.parse_kde_yaml("this is not yaml at all!!!") == {}

    def test_returns_empty_for_empty_string(self):
        assert ex.parse_kde_yaml("") == {}

    def test_parses_fenced_yaml(self):
        fenced = f"```yaml\n{SIMPLE_YAML}```"
        result = ex.parse_kde_yaml(fenced)
        assert "element1" in result

    def test_multiple_elements_renumbered_sequentially(self):
        yaml_text = (
            "element1:\n"
            "  name: Alpha\n"
            "  requirements:\n"
            "    - req a\n"
            "    - req b\n"
            "element2:\n"
            "  name: Beta\n"
            "  requirements:\n"
            "    - req c\n"
            "    - req d\n"
        )
        result = ex.parse_kde_yaml(yaml_text)
        assert set(result.keys()) == {"element1", "element2"}


class TestMergeKdeDicts:
    def test_merges_two_dicts(self):
        a = {"element1": {"name": "A", "requirements": ["r1"]}}
        b = {"element1": {"name": "B", "requirements": ["r2"]}}
        merged = ex.merge_kde_dicts([a, b])
        assert len(merged) == 2
        assert merged["element1"]["name"] == "A"
        assert merged["element2"]["name"] == "B"

    def test_empty_list_returns_empty_dict(self):
        assert ex.merge_kde_dicts([]) == {}

    def test_keys_are_sequential(self):
        dicts = [
            {"element1": {"name": f"X{i}", "requirements": []}}
            for i in range(3)
        ]
        merged = ex.merge_kde_dicts(dicts)
        assert set(merged.keys()) == {"element1", "element2", "element3"}

    def test_single_dict_preserved(self):
        d = {"element1": {"name": "Solo", "requirements": ["only req"]}}
        assert ex.merge_kde_dicts([d]) == {"element1": d["element1"]}


class TestExtractKdesFromPdf:
    """Uses a tiny synthetic PDF and a mocked pipeline – no real model."""

    def test_returns_kdes_from_small_pdf(self, tmp):
        pdf_path = make_pdf(
            str(tmp / "test.pdf"),
            "The kubelet must disable anonymous authentication.",
        )
        pipe = mock_pipe_returning(SIMPLE_YAML)
        result = ex.extract_kdes_from_pdf(
            pdf_path, pipe, ex.build_kde_prompt_zero_shot, max_new_tokens=64
        )
        assert isinstance(result, dict)
        assert len(result) >= 1

    def test_returns_empty_dict_for_missing_pdf(self, tmp, capsys):
        pipe = mock_pipe_returning(SIMPLE_YAML)
        result = ex.extract_kdes_from_pdf(
            str(tmp / "missing.pdf"), pipe, ex.build_kde_prompt_zero_shot
        )
        assert result == {}

    def test_chunk_error_is_skipped_not_raised(self, tmp):
        pdf_path = make_pdf(str(tmp / "test.pdf"), "Some text.")

        def bad_pipe(messages, max_new_tokens=1024):
            raise RuntimeError("simulated model crash")

        result = ex.extract_kdes_from_pdf(
            pdf_path, bad_pipe, ex.build_kde_prompt_zero_shot
        )
        assert isinstance(result, dict)

    def test_all_three_prompt_builders_accepted(self, tmp):
        pdf_path = make_pdf(str(tmp / "test.pdf"), "Kubelet config must be secured.")
        pipe = mock_pipe_returning(SIMPLE_YAML)
        for builder in (
            ex.build_kde_prompt_zero_shot,
            ex.build_kde_prompt_one_shot,
            ex.build_kde_prompt_chain_of_thought,
        ):
            result = ex.extract_kdes_from_pdf(pdf_path, pipe, builder, max_new_tokens=64)
            assert isinstance(result, dict)


class TestProcessTwoFiles:
    """Integration test for the two-file pipeline – mocked pipe, tiny PDFs."""

    def test_produces_two_yaml_files(self, tmp):
        p1 = make_pdf(str(tmp / "doc1.pdf"), "Disable anonymous auth.")
        p2 = make_pdf(str(tmp / "doc2.pdf"), "Enable audit logging.")
        pipe = mock_pipe_returning(SIMPLE_YAML)
        out_dir = str(tmp / "out")

        result = ex.process_two_files(p1, p2, pipe, output_dir=out_dir, max_new_tokens=64)

        assert len(result) == 2
        for key, yaml_path in result.items():
            assert os.path.isfile(yaml_path), f"Missing: {yaml_path}"
            assert yaml_path.endswith("_kdes.yaml")

    def test_output_filenames_are_unique_for_same_input(self, tmp):
        p1 = make_pdf(str(tmp / "same.pdf"), "Content A.")
        pipe = mock_pipe_returning(SIMPLE_YAML)
        out_dir = str(tmp / "out")

        result = ex.process_two_files(p1, p1, pipe, output_dir=out_dir, max_new_tokens=64)
        paths = list(result.values())
        assert paths[0] != paths[1], "Output paths must differ even for identical inputs"
        assert "input1" in paths[0]
        assert "input2" in paths[1]

    def test_result_keys_include_input_index(self, tmp):
        p1 = make_pdf(str(tmp / "a.pdf"), "Text A.")
        p2 = make_pdf(str(tmp / "b.pdf"), "Text B.")
        pipe = mock_pipe_returning(SIMPLE_YAML)
        out_dir = str(tmp / "out")

        result = ex.process_two_files(p1, p2, pipe, output_dir=out_dir, max_new_tokens=64)
        keys = list(result.keys())
        assert any("input1" in k for k in keys)
        assert any("input2" in k for k in keys)

    def test_output_dir_is_created_if_missing(self, tmp):
        p1 = make_pdf(str(tmp / "a.pdf"), "Text.")
        p2 = make_pdf(str(tmp / "b.pdf"), "Text.")
        pipe = mock_pipe_returning(SIMPLE_YAML)
        out_dir = str(tmp / "new_dir" / "nested")

        ex.process_two_files(p1, p2, pipe, output_dir=out_dir, max_new_tokens=64)
        assert os.path.isdir(out_dir)

    def test_yaml_files_are_valid_yaml(self, tmp):
        p1 = make_pdf(str(tmp / "a.pdf"), "Text A.")
        p2 = make_pdf(str(tmp / "b.pdf"), "Text B.")
        pipe = mock_pipe_returning(SIMPLE_YAML)
        out_dir = str(tmp / "out")

        result = ex.process_two_files(p1, p2, pipe, output_dir=out_dir, max_new_tokens=64)
        for yaml_path in result.values():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            assert isinstance(data, (dict, type(None)))


# =============================================================================
# KDE_COMPARE – Task 2
# =============================================================================

def write_kde_yaml(path, kde_dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(kde_dict, f)


KDE_A = {
    "element1": {"name": "Anonymous Auth",
                 "requirements": ["Disable anonymous access", "Enforce auth"]},
    "element2": {"name": "Audit Logging",
                 "requirements": ["Enable audit logs", "Retain for 90 days"]},
    "element3": {"name": "RBAC Roles",
                 "requirements": ["Restrict cluster-admin", "Avoid wildcards"]},
}

KDE_B = {
    "element1": {"name": "Anonymous Auth",
                 "requirements": ["Disable anonymous access", "Review quarterly"]},
    "element2": {"name": "Audit Logging",
                 "requirements": ["Enable audit logs", "Retain for 90 days"]},
    "element3": {"name": "TLS Config",
                 "requirements": ["Use TLS 1.2+", "Rotate certificates"]},
}


class TestLoadYamlOutputs:
    def test_finds_two_yaml_files(self, tmp):
        write_kde_yaml(str(tmp / "input1-cis-r1_kdes.yaml"), KDE_A)
        write_kde_yaml(str(tmp / "input2-cis-r2_kdes.yaml"), KDE_B)
        p1, p2 = cmp.load_yaml_outputs(str(tmp))
        assert "input1" in p1
        assert "input2" in p2

    def test_raises_for_missing_directory(self):
        with pytest.raises(FileNotFoundError):
            cmp.load_yaml_outputs("/no/such/dir")

    def test_raises_when_fewer_than_two_files(self, tmp):
        write_kde_yaml(str(tmp / "input1-only_kdes.yaml"), KDE_A)
        with pytest.raises(FileNotFoundError):
            cmp.load_yaml_outputs(str(tmp))

    def test_returns_input1_before_input2(self, tmp):
        write_kde_yaml(str(tmp / "input2-cis-r2_kdes.yaml"), KDE_B)
        write_kde_yaml(str(tmp / "input1-cis-r1_kdes.yaml"), KDE_A)
        p1, p2 = cmp.load_yaml_outputs(str(tmp))
        assert os.path.basename(p1).startswith("input1")
        assert os.path.basename(p2).startswith("input2")


class TestCompareElementNames:
    def test_detects_names_unique_to_each_file(self, tmp):
        p1 = str(tmp / "input1-a_kdes.yaml")
        p2 = str(tmp / "input2-b_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_B)
        out = str(tmp / "name_diff.txt")

        cmp.compare_element_names(p1, p2, out)
        content = open(out).read()

        assert "RBAC Roles" in content       # only in A
        assert "TLS Config" in content       # only in B
        assert "Audit Logging" not in content  # shared → should not appear

    def test_sentinel_when_names_identical(self, tmp):
        p1 = str(tmp / "input1-a_kdes.yaml")
        p2 = str(tmp / "input2-b_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_A)   # same data
        out = str(tmp / "name_diff.txt")

        cmp.compare_element_names(p1, p2, out)
        assert "NO DIFFERENCES" in open(out).read()

    def test_output_labels_use_filenames(self, tmp):
        p1 = str(tmp / "input1-doc_kdes.yaml")
        p2 = str(tmp / "input2-doc_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_B)
        out = str(tmp / "name_diff.txt")

        cmp.compare_element_names(p1, p2, out)
        content = open(out).read()
        assert "input1-doc_kdes.yaml" in content
        assert "input2-doc_kdes.yaml" in content


class TestCompareElementsAndRequirements:
    def test_name_only_in_one_file_gets_na(self, tmp):
        p1 = str(tmp / "input1-a_kdes.yaml")
        p2 = str(tmp / "input2-b_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_B)
        out = str(tmp / "full_diff.txt")

        cmp.compare_elements_and_requirements(p1, p2, out)
        lines = open(out).read().splitlines()

        na_lines = [l for l in lines if l.endswith(",NA")]
        names_in_na = [l.split(",")[0] for l in na_lines]
        assert "RBAC Roles" in names_in_na
        assert "TLS Config" in names_in_na

    def test_shared_name_different_req_has_req_in_tuple(self, tmp):
        p1 = str(tmp / "input1-a_kdes.yaml")
        p2 = str(tmp / "input2-b_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_B)
        out = str(tmp / "full_diff.txt")

        cmp.compare_elements_and_requirements(p1, p2, out)
        content = open(out).read()

        # Anonymous Auth is shared but has one different req each side
        assert "Anonymous Auth" in content
        assert "Enforce auth" in content
        assert "Review quarterly" in content

    def test_identical_reqs_not_reported(self, tmp):
        p1 = str(tmp / "input1-a_kdes.yaml")
        p2 = str(tmp / "input2-b_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_B)
        out = str(tmp / "full_diff.txt")

        cmp.compare_elements_and_requirements(p1, p2, out)
        content = open(out).read()

        # "Audit Logging" is identical in both → must not appear
        assert "Audit Logging" not in content

    def test_sentinel_when_fully_identical(self, tmp):
        p1 = str(tmp / "input1-a_kdes.yaml")
        p2 = str(tmp / "input2-b_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_A)
        out = str(tmp / "full_diff.txt")

        cmp.compare_elements_and_requirements(p1, p2, out)
        assert "NO DIFFERENCES" in open(out).read()

    def test_tuple_has_four_comma_separated_columns(self, tmp):
        p1 = str(tmp / "input1-a_kdes.yaml")
        p2 = str(tmp / "input2-b_kdes.yaml")
        write_kde_yaml(p1, KDE_A)
        write_kde_yaml(p2, KDE_B)
        out = str(tmp / "full_diff.txt")

        cmp.compare_elements_and_requirements(p1, p2, out)
        for line in open(out).read().splitlines():
            assert line.count(",") == 3, f"Expected 3 commas in: {line!r}"


# =============================================================================
# TASK 3
# =============================================================================

def write_file(path, text):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    open(path, "w").write(text)


DIFF_CONTENT = (
    "Anonymous Auth,ABSENT-IN-b.yaml,PRESENT-IN-a.yaml,Disable anonymous access\n"
    "Audit Logging,ABSENT-IN-a.yaml,PRESENT-IN-b.yaml,NA\n"
    "RBAC Roles,ABSENT-IN-b.yaml,PRESENT-IN-a.yaml,Restrict cluster-admin\n"
)

NO_DIFF_CONTENT = "NO DIFFERENCES IN REGARDS TO ELEMENT NAMES OR REQUIREMENTS\n"


class TestLoadTextOutputs:
    def test_finds_both_text_files(self, tmp):
        write_file(str(tmp / "name_diff.txt"), "some diff")
        write_file(str(tmp / "full_diff.txt"), "some diff")
        n, f = t3.load_text_outputs(str(tmp))
        assert n.endswith("name_diff.txt")
        assert f.endswith("full_diff.txt")

    def test_raises_if_name_diff_missing(self, tmp):
        write_file(str(tmp / "full_diff.txt"), "content")
        with pytest.raises(FileNotFoundError):
            t3.load_text_outputs(str(tmp))

    def test_raises_if_full_diff_missing(self, tmp):
        write_file(str(tmp / "name_diff.txt"), "content")
        with pytest.raises(FileNotFoundError):
            t3.load_text_outputs(str(tmp))

    def test_raises_if_directory_missing(self):
        with pytest.raises(FileNotFoundError):
            t3.load_text_outputs("/no/such/dir")


class TestExtractKdeTermsFromDiff:
    def test_extracts_kde_names(self):
        terms = t3._extract_kde_terms_from_diff.__wrapped__ \
            if hasattr(t3._extract_kde_terms_from_diff, "__wrapped__") \
            else t3._extract_kde_terms_from_diff

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(DIFF_CONTENT)
            path = f.name
        try:
            result = t3._extract_kde_terms_from_diff(path)
            assert "Anonymous Auth" in result
            assert "Audit Logging" in result
            assert "RBAC Roles" in result
        finally:
            os.unlink(path)

    def test_extracts_requirement_strings_excluding_na(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(DIFF_CONTENT)
            path = f.name
        try:
            result = t3._extract_kde_terms_from_diff(path)
            assert "Disable anonymous access" in result
            assert "Restrict cluster-admin" in result
            assert "NA" not in result
        finally:
            os.unlink(path)

    def test_returns_empty_for_no_diff_sentinel(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(NO_DIFF_CONTENT)
            path = f.name
        try:
            assert t3._extract_kde_terms_from_diff(path) == []
        finally:
            os.unlink(path)

    def test_deduplicates_terms(self):
        content = (
            "Anonymous Auth,ABSENT-IN-b,PRESENT-IN-a,req one\n"
            "Anonymous Auth,ABSENT-IN-b,PRESENT-IN-a,req two\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            result = t3._extract_kde_terms_from_diff(path)
            assert result.count("Anonymous Auth") == 1
        finally:
            os.unlink(path)


class TestTermsToControlIds:
    def test_maps_anonymous_to_c0069(self):
        ids = t3._terms_to_control_ids(["Anonymous Auth"])
        assert "C-0069" in ids

    def test_maps_audit_to_c0067(self):
        ids = t3._terms_to_control_ids(["Audit Logging"])
        assert "C-0067" in ids

    def test_maps_rbac_to_c0035(self):
        ids = t3._terms_to_control_ids(["RBAC Roles"])
        assert "C-0035" in ids

    def test_maps_secret_to_c0015(self):
        ids = t3._terms_to_control_ids(["Access to secrets"])
        assert "C-0015" in ids

    def test_returns_sorted_list(self):
        ids = t3._terms_to_control_ids(["Anonymous Auth", "Audit Logging"])
        assert ids == sorted(ids)

    def test_returns_empty_for_unrecognised_terms(self):
        ids = t3._terms_to_control_ids(["Completely unrecognised term XYZ"])
        assert ids == []

    def test_case_insensitive_matching(self):
        ids = t3._terms_to_control_ids(["ANONYMOUS AUTH"])
        assert "C-0069" in ids

    def test_no_duplicate_control_ids(self):
        # Two terms that both map to C-0069 → should appear only once
        ids = t3._terms_to_control_ids(["Anonymous Auth", "anonymous access"])
        assert ids.count("C-0069") == 1


class TestMapDifferencesToKubescapeControls:
    def test_writes_control_ids_when_diffs_exist(self, tmp):
        write_file(str(tmp / "name_diff.txt"), "some diff")
        write_file(str(tmp / "full_diff.txt"), DIFF_CONTENT)
        out = str(tmp / "controls.txt")

        t3.map_differences_to_kubescape_controls(
            str(tmp / "name_diff.txt"), str(tmp / "full_diff.txt"), out
        )
        content = open(out).read().strip()
        assert content != t3.NO_DIFF_SENTINEL
        assert "C-0069" in content   # from Anonymous Auth

    def test_writes_sentinel_when_no_diffs(self, tmp):
        write_file(str(tmp / "name_diff.txt"), NO_DIFF_CONTENT)
        write_file(str(tmp / "full_diff.txt"), NO_DIFF_CONTENT)
        out = str(tmp / "controls.txt")

        t3.map_differences_to_kubescape_controls(
            str(tmp / "name_diff.txt"), str(tmp / "full_diff.txt"), out
        )
        assert open(out).read().strip() == t3.NO_DIFF_SENTINEL

    def test_one_control_id_per_line(self, tmp):
        write_file(str(tmp / "name_diff.txt"), "diff")
        write_file(str(tmp / "full_diff.txt"), DIFF_CONTENT)
        out = str(tmp / "controls.txt")

        t3.map_differences_to_kubescape_controls(
            str(tmp / "name_diff.txt"), str(tmp / "full_diff.txt"), out
        )
        lines = [l for l in open(out).read().splitlines() if l.strip()]
        for line in lines:
            # each line should be a single control ID like C-XXXX
            assert "," not in line, f"Multiple IDs on one line: {line!r}"

    def test_creates_output_directory_if_missing(self, tmp):
        write_file(str(tmp / "name_diff.txt"), "diff")
        write_file(str(tmp / "full_diff.txt"), DIFF_CONTENT)
        out = str(tmp / "subdir" / "controls.txt")

        t3.map_differences_to_kubescape_controls(
            str(tmp / "name_diff.txt"), str(tmp / "full_diff.txt"), out
        )
        assert os.path.isfile(out)


class TestRunKubescape:
    """Mock subprocess so kubescape binary is not required."""

    def _make_kubescape_json(self, tmp) -> str:
        """Write a minimal kubescape JSON result file and return its path."""
        data = {
            "resources": [
                {
                    "resourceID": "res-001",
                    "object": {"metadata": {"name": "my-deploy"}},
                }
            ],
            "results": [
                {
                    "resourceID": "res-001",
                    "controls": [
                        {
                            "controlID": "C-0067",
                            "status": {"status": "failed"},
                        }
                    ],
                }
            ],
            "summaryDetails": {
                "controls": {
                    "C-0067": {
                        "name": "Audit logs enabled",
                        "severity": {"severity": "High"},
                        "resourceCounters": {
                            "failedResources": 1,
                            "passedResources": 2,
                            "skippedResources": 0,
                        },
                    }
                }
            },
        }
        path = str(tmp / "results.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def _make_zip(self, tmp) -> str:
        import zipfile
        z = str(tmp / "project-yamls.zip")
        with zipfile.ZipFile(z, "w") as zf:
            zf.writestr("deploy.yaml", "apiVersion: apps/v1\nkind: Deployment\n")
        return z

    def test_returns_dataframe(self, tmp):
        json_path = self._make_kubescape_json(tmp)
        zip_path  = self._make_zip(tmp)

        write_file(str(tmp / "controls.txt"), "C-0067")

        with patch("subprocess.run") as mock_run, \
             patch("tempfile.TemporaryDirectory") as mock_td:

            # Make TemporaryDirectory return the real tmp so our JSON is found
            mock_td.return_value.__enter__ = lambda s: str(tmp)
            mock_td.return_value.__exit__  = MagicMock(return_value=False)
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            df = t3.run_kubescape(str(tmp / "controls.txt"), zip_path)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "FilePath", "Severity", "Control name",
            "Failed resources", "All Resources", "Compliance score",
        ]

    def test_raises_for_missing_zip(self, tmp):
        write_file(str(tmp / "controls.txt"), "C-0067")
        with pytest.raises(FileNotFoundError):
            t3.run_kubescape(str(tmp / "controls.txt"), "/no/such.zip")

    def test_uses_all_controls_for_sentinel(self, tmp):
        json_path = self._make_kubescape_json(tmp)
        zip_path  = self._make_zip(tmp)
        write_file(str(tmp / "controls.txt"), t3.NO_DIFF_SENTINEL)

        with patch("subprocess.run") as mock_run, \
             patch("tempfile.TemporaryDirectory") as mock_td:

            mock_td.return_value.__enter__ = lambda s: str(tmp)
            mock_td.return_value.__exit__  = MagicMock(return_value=False)
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            t3.run_kubescape(str(tmp / "controls.txt"), zip_path)

        cmd = mock_run.call_args[0][0]
        # "control" subcommand must NOT appear when running all controls
        assert "control" not in cmd

    def test_uses_specific_controls_when_listed(self, tmp):
        json_path = self._make_kubescape_json(tmp)
        zip_path  = self._make_zip(tmp)
        write_file(str(tmp / "controls.txt"), "C-0067\nC-0069\n")

        with patch("subprocess.run") as mock_run, \
             patch("tempfile.TemporaryDirectory") as mock_td:

            mock_td.return_value.__enter__ = lambda s: str(tmp)
            mock_td.return_value.__exit__  = MagicMock(return_value=False)
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

            t3.run_kubescape(str(tmp / "controls.txt"), zip_path)

        cmd = mock_run.call_args[0][0]
        assert "control" in cmd
        # The IDs should appear as a comma-joined argument
        joined = " ".join(cmd)
        assert "C-0067" in joined and "C-0069" in joined

    def test_raises_runtime_error_for_kubescape_not_found(self, tmp):
        zip_path = self._make_zip(tmp)
        write_file(str(tmp / "controls.txt"), "C-0067")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="kubescape binary not found"):
                t3.run_kubescape(str(tmp / "controls.txt"), zip_path)


class TestGenerateCsv:
    def _sample_df(self):
        return pd.DataFrame([
            {
                "FilePath": "deploy.yaml",
                "Severity": "High",
                "Control name": "Audit logs enabled",
                "Failed resources": 2,
                "All Resources": 5,
                "Compliance score": "60.0%",
            },
            {
                "FilePath": "service.yaml",
                "Severity": "Medium",
                "Control name": "Anonymous Auth",
                "Failed resources": 1,
                "All Resources": 3,
                "Compliance score": "66.7%",
            },
        ])

    def test_creates_csv_file(self, tmp):
        out = str(tmp / "results.csv")
        t3.generate_csv(self._sample_df(), out)
        assert os.path.isfile(out)

    def test_csv_has_correct_headers(self, tmp):
        out = str(tmp / "results.csv")
        t3.generate_csv(self._sample_df(), out)
        headers = next(csv.reader(open(out)))
        assert headers == [
            "FilePath", "Severity", "Control name",
            "Failed resources", "All Resources", "Compliance score",
        ]

    def test_csv_row_count_matches_dataframe(self, tmp):
        df = self._sample_df()
        out = str(tmp / "results.csv")
        t3.generate_csv(df, out)
        rows = list(csv.DictReader(open(out)))
        assert len(rows) == len(df)

    def test_csv_values_match_dataframe(self, tmp):
        out = str(tmp / "results.csv")
        t3.generate_csv(self._sample_df(), out)
        rows = list(csv.DictReader(open(out)))
        assert rows[0]["FilePath"] == "deploy.yaml"
        assert rows[0]["Severity"] == "High"
        assert rows[1]["Compliance score"] == "66.7%"

    def test_missing_columns_filled_with_empty_string(self, tmp):
        df = pd.DataFrame([{"FilePath": "f.yaml", "Severity": "Low"}])
        out = str(tmp / "results.csv")
        t3.generate_csv(df, out)
        row = next(csv.DictReader(open(out)))
        assert row["Control name"] == ""
        assert row["Failed resources"] == ""

    def test_creates_output_directory_if_missing(self, tmp):
        out = str(tmp / "subdir" / "deep" / "results.csv")
        t3.generate_csv(self._sample_df(), out)
        assert os.path.isfile(out)