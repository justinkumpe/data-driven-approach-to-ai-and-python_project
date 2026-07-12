#!/usr/bin/env python3
"""Grade the Udacity PCEP ecommerce project against the official rubric.

Exit codes:
  0 — all required checks passed
  1 — one or more required checks failed
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_CANDIDATES = [
    ROOT / "Udacity - PCEP Project.ipynb",
    *sorted(ROOT.glob("*.ipynb")),
]
USER_INPUTS = ROOT / "user_inputs.txt"
TOP_PRODUCTS_CSV = ROOT / "top_products.csv"
TOP_PRODUCTS_PARQUET = ROOT / "top_products.parquet"


@dataclass
class CheckResult:
    criterion: str
    passed: bool
    detail: str
    required: bool = True


@dataclass
class GradeReport:
    results: list[CheckResult] = field(default_factory=list)

    def add(
        self,
        criterion: str,
        passed: bool,
        detail: str,
        *,
        required: bool = True,
    ) -> None:
        self.results.append(
            CheckResult(criterion, passed, detail, required=required)
        )

    @property
    def required_failures(self) -> list[CheckResult]:
        return [r for r in self.results if r.required and not r.passed]

    @property
    def all_required_passed(self) -> bool:
        return not self.required_failures

    def to_markdown(self) -> str:
        lines = [
            "# PCEP Project Rubric Grade",
            "",
            f"**Overall:** {'PASS' if self.all_required_passed else 'FAIL'}",
            "",
            "| Status | Criterion | Detail |",
            "|--------|-----------|--------|",
        ]
        for r in self.results:
            icon = "✅" if r.passed else ("❌" if r.required else "⚠️")
            detail = r.detail.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {icon} | {r.criterion} | {detail} |")

        failed = self.required_failures
        if failed:
            lines.extend(["", "## Failed required checks", ""])
            for r in failed:
                lines.append(f"- **{r.criterion}**: {r.detail}")
        return "\n".join(lines) + "\n"


def find_notebook() -> Path | None:
    for path in NOTEBOOK_CANDIDATES:
        if path.is_file() and path.name != ".ipynb_checkpoints":
            return path
    return None


def notebook_source(nb: dict) -> str:
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            chunks.append("".join(cell.get("source", [])))
    return "\n".join(chunks)


def has_pattern(source: str, pattern: str, flags: int = re.MULTILINE) -> bool:
    return re.search(pattern, source, flags) is not None


def code_cells_executed(nb: dict) -> tuple[int, int, int]:
    """Return (code_cells, executed_cells, error_outputs)."""
    code_cells = 0
    executed = 0
    errors = 0
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code_cells += 1
        if cell.get("execution_count") is not None:
            executed += 1
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error" or output.get("ename"):
                errors += 1
    return code_cells, executed, errors


def count_plot_outputs(nb: dict) -> int:
    count = 0
    for cell in nb.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                count += 1
                break
    return count


def check_env(report: GradeReport) -> None:
    report.add(
        "1.1 Python installed",
        True,
        f"Python {sys.version.split()[0]}",
    )

    missing: list[str] = []
    versions: list[str] = []
    for mod_name, import_name in [
        ("datasets", "datasets"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("pyarrow", "pyarrow"),
    ]:
        try:
            mod = __import__(import_name)
            versions.append(f"{mod_name}={getattr(mod, '__version__', '?')}")
        except Exception:
            missing.append(mod_name)

    report.add(
        "1.2 Required libraries installed",
        not missing,
        (
            f"Missing: {', '.join(missing)}"
            if missing
            else ", ".join(versions)
        ),
    )


def check_reviews_loading(source: str, report: GradeReport) -> None:
    report.add(
        "2.2 sample_limit = 100",
        has_pattern(source, r"sample_limit\s*=\s*100\b"),
        "Found sample_limit = 100"
        if has_pattern(source, r"sample_limit\s*=\s*100\b")
        else "Expected sample_limit = 100",
    )
    report.add(
        "2.2 reviews for-loop with break",
        has_pattern(
            source,
            r"for\s+\w+,\s*\w+\s+in\s+enumerate\(\s*reviews_dataset\s*\).*?if\s+\w+\s*>=\s*sample_limit\s*:.*?break",
            re.DOTALL,
        ),
        "Found enumerate loop that breaks at sample_limit",
    )
    report.add(
        "2.2 load_dataset streaming=True",
        has_pattern(source, r"streaming\s*=\s*True"),
        "streaming=True present",
    )
    report.add(
        "2.2 load_dataset trust_remote_code=True",
        has_pattern(source, r"trust_remote_code\s*=\s*True"),
        "trust_remote_code=True present",
    )
    report.add(
        "2.2 reviews DataFrame created",
        has_pattern(source, r"reviews_df\s*=\s*pd\.DataFrame"),
        "reviews_df = pd.DataFrame(...)",
    )


def check_inspect_print(source: str, report: GradeReport) -> None:
    report.add(
        "2.3 reviews_df.head()",
        has_pattern(source, r"reviews_df\.head\s*\("),
        "reviews_df.head() used",
    )
    report.add(
        "2.4 Items_To_Print = 10",
        has_pattern(source, r"Items_To_Print\s*=\s*10\b"),
        "Items_To_Print = 10",
    )
    report.add(
        "2.4 head(Items_To_Print)",
        has_pattern(source, r"reviews_df\.head\s*\(\s*Items_To_Print\s*\)"),
        "prints first Items_To_Print rows",
    )


def check_metadata(source: str, report: GradeReport) -> None:
    report.add(
        "3.1 metadata_limit = 100",
        has_pattern(source, r"metadata_limit\s*=\s*100\b"),
        "metadata_limit = 100",
    )
    report.add(
        "3.1 metadata for-loop with break",
        has_pattern(
            source,
            r"for\s+\w+,\s*\w+\s+in\s+enumerate\(\s*item_metadata_dataset\s*\).*?if\s+\w+\s*>=\s*metadata_limit\s*:.*?break",
            re.DOTALL,
        ),
        "Found enumerate loop that breaks at metadata_limit",
    )
    report.add(
        "3.1 item_metadata_df created",
        has_pattern(source, r"item_metadata_df\s*=\s*pd\.DataFrame"),
        "item_metadata_df = pd.DataFrame(...)",
    )
    report.add(
        "3.2 item_metadata_df.head()",
        has_pattern(source, r"item_metadata_df\.head\s*\("),
        "item_metadata_df.head() used",
    )


def check_get_product(source: str, report: GradeReport) -> None:
    has_fn = has_pattern(source, r"def\s+get_product\s*\(")
    has_try = has_pattern(
        source,
        r"def\s+get_product\s*\(.*?\):\s*.*?try\s*:.*?except\s+",
        re.DOTALL,
    )
    report.add(
        "3.3 get_product with try/except",
        has_fn and has_try,
        "get_product defines try/except exception handling"
        if has_fn and has_try
        else "Expected get_product with try/except",
    )


def check_columns_info(source: str, report: GradeReport) -> None:
    report.add(
        "4.1 reviews_df.columns / item_metadata_df.columns",
        has_pattern(source, r"reviews_df\.columns")
        and has_pattern(source, r"item_metadata_df\.columns"),
        "Both DataFrame .columns inspected",
    )
    report.add(
        "4.1 for-loop over .columns",
        has_pattern(source, r"for\s+\w+\s+in\s+reviews_df\.columns")
        and has_pattern(source, r"for\s+\w+\s+in\s+item_metadata_df\.columns"),
        "for loops iterate column names as a list",
    )
    report.add(
        "4.2 .dtypes reviewed",
        has_pattern(source, r"reviews_df\.dtypes")
        and has_pattern(source, r"item_metadata_df\.dtypes"),
        "Both DataFrame .dtypes printed",
    )
    report.add(
        "4.3 .info() summaries",
        has_pattern(source, r"reviews_df\.info\s*\(")
        and has_pattern(source, r"item_metadata_df\.info\s*\("),
        "Both DataFrame .info() called",
    )


def check_cleaning(source: str, report: GradeReport) -> None:
    report.add(
        "5.1 average_rating_filter = 4.5",
        has_pattern(source, r"average_rating_filter\s*=\s*4\.5\b"),
        "average_rating_filter is float 4.5",
    )
    report.add(
        "5.1 item_metadata_cleaned_df columns",
        has_pattern(
            source,
            r"item_metadata_cleaned_df\s*=\s*item_metadata_df\s*\[\s*\[[^\]]*title[^\]]*average_rating[^\]]*price[^\]]*\]\s*\]",
        )
        or (
            has_pattern(source, r"item_metadata_cleaned_df\s*=")
            and has_pattern(source, r"\[[^\]]*['\"]title['\"][^\]]*['\"]average_rating['\"][^\]]*['\"]price['\"]")
        ),
        "Cleaned DF includes title, average_rating, price",
    )
    report.add(
        "5.1 top_products_df filters",
        has_pattern(source, r"top_products_df\s*=")
        and has_pattern(source, r"average_rating['\"]\s*\]\s*>=\s*average_rating_filter")
        and has_pattern(source, r"price['\"]\s*\]\.notna\s*\(")
        and has_pattern(source, r"title['\"]\s*\]\.notna\s*\("),
        "Filters title present, rating >= filter, price present",
    )


def check_percentage(source: str, report: GradeReport) -> None:
    has_fn = has_pattern(source, r"def\s+calculate_percentage\s*\(")
    has_return = has_pattern(
        source,
        r"def\s+calculate_percentage\s*\(.*?\):.*?return\s+",
        re.DOTALL,
    )
    has_math = has_pattern(
        source,
        r"def\s+calculate_percentage\s*\(.*?\):.*?(part\s*/\s*whole|/\s*whole).*?\*\s*100",
        re.DOTALL,
    )
    report.add(
        "5.3 calculate_percentage returns part/whole",
        has_fn and has_return and has_math,
        "Function returns (part / whole) * 100"
        if has_fn and has_return and has_math
        else "Expected calculate_percentage with return and part/whole math",
    )


def check_exports_in_notebook(source: str, report: GradeReport) -> None:
    report.add(
        "5.4 to_csv('top_products.csv')",
        has_pattern(source, r"\.to_csv\s*\(\s*['\"]top_products\.csv['\"]"),
        "Notebook saves top_products.csv",
    )
    report.add(
        "5.5 to_parquet('top_products.parquet')",
        has_pattern(source, r"\.to_parquet\s*\(\s*['\"]top_products\.parquet['\"]"),
        "Notebook saves top_products.parquet",
    )


def check_top_rated_titles(source: str, report: GradeReport) -> None:
    report.add(
        "5.6 top_rated_titles list + append loop",
        has_pattern(source, r"top_rated_titles\s*=\s*\[\s*\]")
        and has_pattern(
            source,
            r"for\s+\w+\s+in\s+top_products_df\s*\[\s*['\"]title['\"]\s*\]\s*:.*?top_rated_titles\.append",
            re.DOTALL,
        )
        and has_pattern(source, r"print\s*\(\s*top_rated_titles\s*\)"),
        "Empty list, append titles in a for loop, and print results",
    )


def check_visualizations(source: str, report: GradeReport) -> None:
    report.add(
        "6.1 histplot of top_products_df['price'] with kde=True",
        has_pattern(source, r"sns\.histplot\s*\(")
        and has_pattern(source, r"top_products_df\s*\[\s*['\"]price['\"]\s*\]")
        and has_pattern(source, r"kde\s*=\s*True"),
        "Histogram uses price column and kde=True",
    )
    report.add(
        "6.2 scatterplot price vs average_rating",
        has_pattern(
            source,
            r"sns\.scatterplot\s*\([\s\S]*?data\s*=\s*top_products_df[\s\S]*?x\s*=\s*['\"]price['\"][\s\S]*?y\s*=\s*['\"]average_rating['\"]",
        )
        or has_pattern(
            source,
            r"sns\.scatterplot\s*\([\s\S]*?x\s*=\s*['\"]price['\"][\s\S]*?y\s*=\s*['\"]average_rating['\"][\s\S]*?data\s*=\s*top_products_df",
        ),
        "Scatterplot uses top_products_df with x=price, y=average_rating",
    )
    report.add(
        "6.4 average_rating cutoff = 4.7",
        has_pattern(source, r"(average_rating_cutoff|highly_rated_products_df_cutoff)\s*=\s*4\.7\b"),
        "Cutoff threshold set to float 4.7",
    )
    report.add(
        "6.5 sorted highly_rated_products_df + scatterplot",
        has_pattern(source, r"highly_rated_products_df_cutoff\s*=\s*4\.7\b")
        and has_pattern(
            source,
            r"highly_rated_products_df\s*=\s*highly_rated_products_df\.sort_values\s*\(",
        )
        and has_pattern(
            source,
            r"sns\.scatterplot\s*\([\s\S]*?data\s*=\s*highly_rated_products_df",
        ),
        "Cutoff, sort_values, and dashboard scatterplot present",
    )


def check_notebook_outputs(nb: dict, report: GradeReport) -> None:
    code_cells, executed, errors = code_cells_executed(nb)
    plots = count_plot_outputs(nb)
    report.add(
        "Deliverable: notebook cell outputs included",
        code_cells > 0 and executed >= max(1, int(code_cells * 0.8)) and errors == 0,
        f"{executed}/{code_cells} code cells executed, {errors} errors, {plots} plot output(s)",
    )
    report.add(
        "Deliverable: visualization outputs present",
        plots >= 2,
        f"Found {plots} cells with plot images (need at least 2)",
    )


def check_user_inputs(report: GradeReport) -> None:
    if not USER_INPUTS.is_file():
        report.add(
            "Deliverable: user_inputs.txt",
            False,
            "user_inputs.txt is missing",
        )
        return

    text = USER_INPUTS.read_text(encoding="utf-8", errors="replace")
    questions = re.findall(r"^Question\s+(\d+)\s*:", text, flags=re.MULTILINE)
    has_name = bool(re.search(r"Completed by:\s*\S+", text))
    nonempty_answers = 0
    for match in re.finditer(
        r"^Question\s+\d+\s*:\s*(.+)$", text, flags=re.MULTILINE
    ):
        if match.group(1).strip():
            nonempty_answers += 1

    report.add(
        "Deliverable: user_inputs.txt answers",
        has_name and nonempty_answers >= 7 and len(questions) >= 7,
        f"name={'yes' if has_name else 'no'}, answered questions={nonempty_answers}",
    )


def check_dataset_files(report: GradeReport) -> None:
    if not TOP_PRODUCTS_CSV.is_file():
        report.add("Deliverable: top_products.csv", False, "File missing")
    else:
        try:
            import pandas as pd

            df = pd.read_csv(TOP_PRODUCTS_CSV)
            required_cols = {"title", "average_rating", "price"}
            ok_cols = required_cols.issubset(df.columns)
            ok_rating = (
                len(df) > 0 and float(df["average_rating"].min()) >= 4.5
            )
            ok_price = df["price"].notna().all()
            ok_title = df["title"].notna().all()
            report.add(
                "Deliverable: top_products.csv",
                ok_cols and ok_rating and ok_price and ok_title,
                f"rows={len(df)}, cols={list(df.columns)}, "
                f"min_rating={float(df['average_rating'].min()) if len(df) else 'n/a'}",
            )
        except Exception as exc:
            report.add(
                "Deliverable: top_products.csv",
                False,
                f"Could not validate CSV: {exc}",
            )

    if not TOP_PRODUCTS_PARQUET.is_file():
        report.add(
            "Deliverable: top_products.parquet",
            False,
            "File missing",
        )
    else:
        try:
            import pandas as pd

            pq = pd.read_parquet(TOP_PRODUCTS_PARQUET)
            report.add(
                "Deliverable: top_products.parquet",
                len(pq) > 0 and {"title", "average_rating", "price"}.issubset(pq.columns),
                f"rows={len(pq)}, cols={list(pq.columns)}",
            )
        except Exception as exc:
            report.add(
                "Deliverable: top_products.parquet",
                False,
                f"Could not validate Parquet: {exc}",
            )


def grade() -> GradeReport:
    report = GradeReport()
    check_env(report)

    notebook_path = find_notebook()
    if notebook_path is None:
        report.add(
            "Notebook present",
            False,
            "No .ipynb notebook found in project root",
        )
        check_user_inputs(report)
        check_dataset_files(report)
        return report

    report.add("Notebook present", True, notebook_path.name)

    try:
        nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add("Notebook readable JSON", False, str(exc))
        check_user_inputs(report)
        check_dataset_files(report)
        return report

    source = notebook_source(nb)
    check_reviews_loading(source, report)
    check_inspect_print(source, report)
    check_metadata(source, report)
    check_get_product(source, report)
    check_columns_info(source, report)
    check_cleaning(source, report)
    check_percentage(source, report)
    check_exports_in_notebook(source, report)
    check_top_rated_titles(source, report)
    check_visualizations(source, report)
    check_notebook_outputs(nb, report)
    check_user_inputs(report)
    check_dataset_files(report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Write a markdown report to this path",
    )
    parser.add_argument(
        "--github-summary",
        action="store_true",
        help="Also append the markdown report to $GITHUB_STEP_SUMMARY",
    )
    args = parser.parse_args(argv)

    report = grade()
    md = report.to_markdown()
    print(md)

    if args.markdown:
        args.markdown.write_text(md, encoding="utf-8")

    if args.github_summary:
        summary_file = os.environ.get("GITHUB_STEP_SUMMARY", "")
        if summary_file:
            try:
                with open(summary_file, "a", encoding="utf-8") as handle:
                    handle.write(md)
            except OSError as exc:
                print(f"Warning: could not write GitHub step summary: {exc}")

    passed = sum(1 for r in report.results if r.passed)
    total = len(report.results)
    print(f"Score: {passed}/{total} checks passed")
    if report.required_failures:
        print(f"Required failures: {len(report.required_failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
