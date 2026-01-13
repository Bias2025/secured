#!/usr/bin/env python3
"""
Comprehensive Garak LLM Security Scanner Toolkit
===============================================

A toolkit for conducting security assessments of LLMs using the Garak vulnerability scanner.

Key Streamlit Cloud hardening:
- Uses the *current* interpreter (sys.executable) instead of `python3`
  so subprocess calls run inside the same venv Streamlit installed into.
- Safer output directory handling + path normalization
- Less fragile logging defaults (writes logs into output_dir when possible)

License: Apache 2.0
Requirements: garak, python>=3.10, PyYAML
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# ----------------------------
# Logging
# ----------------------------

logger = logging.getLogger("garak_scanner_toolkit")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


# ----------------------------
# Enums / Dataclasses
# ----------------------------

class ScanCategory(Enum):
    """Security scan categories based on threat types"""
    JAILBREAKS = "jailbreaks"
    PROMPT_INJECTION = "prompt_injection"
    DATA_LEAKAGE = "data_leakage"
    TOXICITY = "toxicity"
    MALWARE = "malware"
    HALLUCINATION = "hallucination"
    ENCODING = "encoding"
    COMPREHENSIVE = "comprehensive"
    CUSTOM = "custom"
    TEST = "test"


class SeverityLevel(Enum):
    """Vulnerability severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class ProbeConfig:
    """Configuration for individual probes"""
    name: str
    category: str
    description: str
    severity: SeverityLevel
    enabled: bool = True
    custom_params: Optional[Dict] = None


@dataclass
class ScanConfig:
    """Comprehensive scan configuration"""
    target_model: str
    model_type: str
    scan_categories: List[ScanCategory]
    output_dir: str
    report_prefix: str
    max_generations: int = 10
    timeout: int = 3600
    parallel_probes: int = 1
    custom_probes: Optional[List[str]] = None
    exclude_probes: Optional[List[str]] = None
    api_keys: Optional[Dict[str, str]] = None


# ----------------------------
# Probe Registry
# ----------------------------

class GarakProbeRegistry:
    """Registry of Garak probes organized by category"""

    def __init__(self) -> None:
        self.probes = self._initialize_probes()

    def _initialize_probes(self) -> Dict[str, List[ProbeConfig]]:
        """Initialize a probe registry (static list)"""
        return {
            "jailbreaks": [
                ProbeConfig("dan.Dan_11_0", "jailbreaks", "DAN 11.0 jailbreak attempt", SeverityLevel.HIGH),
                ProbeConfig("dan.Dan_10_0", "jailbreaks", "DAN 10.0 jailbreak attempt", SeverityLevel.HIGH),
                ProbeConfig("dan.Dan_9_0", "jailbreaks", "DAN 9.0 jailbreak attempt", SeverityLevel.HIGH),
                ProbeConfig("dan.Dan_8_0", "jailbreaks", "DAN 8.0 jailbreak attempt", SeverityLevel.HIGH),
                ProbeConfig("dan.Dan_7_0", "jailbreaks", "DAN 7.0 jailbreak attempt", SeverityLevel.HIGH),
                ProbeConfig("dan.Dan_6_2", "jailbreaks", "DAN 6.2 jailbreak attempt", SeverityLevel.HIGH),
                ProbeConfig("dan.Dan_6_0", "jailbreaks", "DAN 6.0 jailbreak attempt", SeverityLevel.HIGH),
                ProbeConfig("dan.AntiDAN", "jailbreaks", "Anti-DAN jailbreak technique", SeverityLevel.HIGH),
                ProbeConfig("dan.AutoDAN", "jailbreaks", "Automated DAN jailbreak", SeverityLevel.CRITICAL),
                ProbeConfig("dan.AutoDANCached", "jailbreaks", "Cached AutoDAN attacks", SeverityLevel.CRITICAL),
                ProbeConfig("dan.ChatGPT_Developer_Mode_v2", "jailbreaks", "Developer Mode jailbreak v2", SeverityLevel.HIGH),
                ProbeConfig("dan.ChatGPT_Developer_Mode_RANTI", "jailbreaks", "RANTI Developer Mode jailbreak", SeverityLevel.HIGH),
                ProbeConfig("dan.ChatGPT_Image_Markdown", "jailbreaks", "Image markdown jailbreak", SeverityLevel.MEDIUM),
                ProbeConfig("dan.DAN_Jailbreak", "jailbreaks", "Generic DAN jailbreak", SeverityLevel.HIGH),
                ProbeConfig("dan.DUDE", "jailbreaks", "DUDE jailbreak technique", SeverityLevel.HIGH),
                ProbeConfig("dan.DanInTheWild", "jailbreaks", "Real-world DAN variations", SeverityLevel.HIGH),
                ProbeConfig("dan.DanInTheWildMini", "jailbreaks", "Mini real-world DAN variations", SeverityLevel.MEDIUM),
                ProbeConfig("dan.STAN", "jailbreaks", "STAN jailbreak technique", SeverityLevel.HIGH),
                ProbeConfig("dan.Ablation_Dan_11_0", "jailbreaks", "DAN 11.0 ablation study", SeverityLevel.MEDIUM),
                ProbeConfig("tap.TAP", "jailbreaks", "Tree of Attacks with Pruning", SeverityLevel.CRITICAL),
                ProbeConfig("tap.TAPCached", "jailbreaks", "Cached TAP attacks", SeverityLevel.CRITICAL),
                ProbeConfig("tap.PAIR", "jailbreaks", "PAIR jailbreak technique", SeverityLevel.HIGH),
                ProbeConfig("goodside", "jailbreaks", "Riley Goodside jailbreak collection", SeverityLevel.HIGH),
                ProbeConfig("grandma", "jailbreaks", "Grandma exploit technique", SeverityLevel.MEDIUM),
            ],
            "prompt_injection": [
                ProbeConfig("promptinject.HijackHateHumans", "prompt_injection", "Hate humans injection", SeverityLevel.CRITICAL),
                ProbeConfig("promptinject.HijackHateHumansMini", "prompt_injection", "Mini hate humans injection", SeverityLevel.HIGH),
                ProbeConfig("promptinject.HijackKillHumans", "prompt_injection", "Kill humans injection", SeverityLevel.CRITICAL),
                ProbeConfig("promptinject.HijackKillHumansMini", "prompt_injection", "Mini kill humans injection", SeverityLevel.CRITICAL),
                ProbeConfig("promptinject.HijackLongPrompt", "prompt_injection", "Long prompt hijacking", SeverityLevel.HIGH),
                ProbeConfig("promptinject.HijackLongPromptMini", "prompt_injection", "Mini long prompt hijacking", SeverityLevel.MEDIUM),
                ProbeConfig("latentinjection", "prompt_injection", "Latent injection attacks", SeverityLevel.HIGH),
            ],
            "encoding": [
                ProbeConfig("encoding.InjectBase64", "encoding", "Base64 encoding injection", SeverityLevel.HIGH),
                ProbeConfig("encoding.InjectBase32", "encoding", "Base32 encoding injection", SeverityLevel.HIGH),
                ProbeConfig("encoding.InjectBase16", "encoding", "Base16/Hex encoding injection", SeverityLevel.HIGH),
                ProbeConfig("encoding.InjectAscii85", "encoding", "ASCII85 encoding injection", SeverityLevel.MEDIUM),
                ProbeConfig("encoding.InjectBase2048", "encoding", "Base2048 encoding injection", SeverityLevel.MEDIUM),
                ProbeConfig("encoding.InjectBraille", "encoding", "Braille encoding injection", SeverityLevel.MEDIUM),
                ProbeConfig("encoding.InjectHex", "encoding", "Hexadecimal encoding injection", SeverityLevel.HIGH),
                ProbeConfig("encoding.InjectMime", "encoding", "MIME encoding injection", SeverityLevel.HIGH),
                ProbeConfig("encoding.InjectMorse", "encoding", "Morse code injection", SeverityLevel.LOW),
                ProbeConfig("encoding.InjectQP", "encoding", "Quoted-printable injection", SeverityLevel.MEDIUM),
                ProbeConfig("encoding.InjectROT13", "encoding", "ROT13 encoding injection", SeverityLevel.LOW),
                ProbeConfig("encoding.InjectUU", "encoding", "UUencoding injection", SeverityLevel.MEDIUM),
            ],
            "toxicity": [
                ProbeConfig("atkgen.Tox", "toxicity", "Automated toxicity generation", SeverityLevel.CRITICAL),
                ProbeConfig("realtoxicityprompts", "toxicity", "Real toxicity prompts dataset", SeverityLevel.HIGH),
                ProbeConfig("lmrc.Profanity", "toxicity", "Profanity generation test", SeverityLevel.MEDIUM),
                ProbeConfig("lmrc.Bullying", "toxicity", "Bullying content generation", SeverityLevel.HIGH),
                ProbeConfig("lmrc.SexualContent", "toxicity", "Sexual content generation", SeverityLevel.HIGH),
                ProbeConfig("lmrc.Sexualisation", "toxicity", "Sexualization content", SeverityLevel.HIGH),
                ProbeConfig("lmrc.SlurUsage", "toxicity", "Slur usage detection", SeverityLevel.HIGH),
                ProbeConfig("lmrc.Deadnaming", "toxicity", "Deadnaming behavior", SeverityLevel.MEDIUM),
                ProbeConfig("continuation.ContinueSlursReclaimedSlurs", "toxicity", "Slur continuation test", SeverityLevel.HIGH),
                ProbeConfig("continuation.ContinueSlursReclaimedSlursMini", "toxicity", "Mini slur continuation", SeverityLevel.MEDIUM),
            ],
            "malware": [
                ProbeConfig("malwaregen.TopLevel", "malware", "High-level malware generation", SeverityLevel.CRITICAL),
                ProbeConfig("malwaregen.Payload", "malware", "Malware payload generation", SeverityLevel.CRITICAL),
                ProbeConfig("malwaregen.SubFunctions", "malware", "Malware subfunctions", SeverityLevel.HIGH),
                ProbeConfig("malwaregen.Evasion", "malware", "Anti-detection evasion code", SeverityLevel.HIGH),
            ],
            "data_leakage": [
                ProbeConfig("leakreplay.GuardianCloze", "data_leakage", "Guardian dataset cloze test", SeverityLevel.HIGH),
                ProbeConfig("leakreplay.GuardianComplete", "data_leakage", "Guardian dataset completion", SeverityLevel.HIGH),
                ProbeConfig("leakreplay.LiteratureCloze", "data_leakage", "Literature cloze test", SeverityLevel.MEDIUM),
                ProbeConfig("leakreplay.LiteratureCloze80", "data_leakage", "Literature cloze 80% test", SeverityLevel.MEDIUM),
                ProbeConfig("leakreplay.LiteratureComplete", "data_leakage", "Literature completion test", SeverityLevel.MEDIUM),
                ProbeConfig("leakreplay.LiteratureComplete80", "data_leakage", "Literature completion 80%", SeverityLevel.MEDIUM),
                ProbeConfig("leakreplay.NYTCloze", "data_leakage", "NYT articles cloze test", SeverityLevel.HIGH),
                ProbeConfig("leakreplay.NYTComplete", "data_leakage", "NYT articles completion", SeverityLevel.HIGH),
                ProbeConfig("leakreplay.PotterCloze", "data_leakage", "Harry Potter cloze test", SeverityLevel.LOW),
                ProbeConfig("leakreplay.PotterComplete", "data_leakage", "Harry Potter completion", SeverityLevel.LOW),
            ],
            "hallucination": [
                ProbeConfig("snowball", "hallucination", "Snowball hallucination test", SeverityLevel.MEDIUM),
                ProbeConfig("packagehallucination.Python", "hallucination", "Python package hallucination", SeverityLevel.HIGH),
                ProbeConfig("packagehallucination.JavaScript", "hallucination", "JavaScript package hallucination", SeverityLevel.HIGH),
                ProbeConfig("packagehallucination.Ruby", "hallucination", "Ruby gem hallucination", SeverityLevel.HIGH),
                ProbeConfig("packagehallucination.Rust", "hallucination", "Rust crate hallucination", SeverityLevel.HIGH),
                ProbeConfig("misleading.FalseAssertion50", "hallucination", "False assertions test", SeverityLevel.MEDIUM),
            ],
            "specialized": [
                ProbeConfig("xss", "specialized", "Cross-site scripting vulnerabilities", SeverityLevel.CRITICAL),
                ProbeConfig("donotanswer.DiscriminationExclusionToxicityHatefulOffensive", "specialized", "Discrimination content", SeverityLevel.HIGH),
                ProbeConfig("donotanswer.HumanChatbox", "specialized", "Human chatbox simulation", SeverityLevel.MEDIUM),
                ProbeConfig("donotanswer.InformationHazard", "specialized", "Information hazards", SeverityLevel.HIGH),
                ProbeConfig("donotanswer.MaliciousUses", "specialized", "Malicious use cases", SeverityLevel.HIGH),
                ProbeConfig("donotanswer.MisinformationHarms", "specialized", "Misinformation harms", SeverityLevel.HIGH),
                ProbeConfig("lmrc.Anthropomorphisation", "specialized", "Anthropomorphization issues", SeverityLevel.LOW),
                ProbeConfig("lmrc.QuackMedicine", "specialized", "Quack medicine content", SeverityLevel.HIGH),
                ProbeConfig("divergence.Repeat", "specialized", "Repetition divergence", SeverityLevel.LOW),
                ProbeConfig("divergence.RepeatExtended", "specialized", "Extended repetition test", SeverityLevel.LOW),
                ProbeConfig("phrasing.FutureTense", "specialized", "Future tense manipulation", SeverityLevel.LOW),
                ProbeConfig("phrasing.PastTense", "specialized", "Past tense manipulation", SeverityLevel.LOW),
                ProbeConfig("phrasing.TenseMini", "specialized", "Tense manipulation mini", SeverityLevel.LOW),
                ProbeConfig("suffix", "specialized", "Suffix-based attacks", SeverityLevel.MEDIUM),
                ProbeConfig("topic", "specialized", "Topic-based probes", SeverityLevel.MEDIUM),
                ProbeConfig("visual_jailbreak", "specialized", "Visual jailbreak attempts", SeverityLevel.HIGH),
                ProbeConfig("fileformats", "specialized", "File format exploits", SeverityLevel.MEDIUM),
                ProbeConfig("exploitation", "specialized", "General exploitation techniques", SeverityLevel.HIGH),
                ProbeConfig("glitch", "specialized", "Glitch-based attacks", SeverityLevel.MEDIUM),
                ProbeConfig("ansiescape", "specialized", "ANSI escape sequence attacks", SeverityLevel.LOW),
            ],
            "antivirus_spam": [
                ProbeConfig("av_spam_scanning.EICAR", "antivirus_spam", "EICAR test string", SeverityLevel.LOW),
                ProbeConfig("av_spam_scanning.GTUBE", "antivirus_spam", "GTUBE spam test", SeverityLevel.LOW),
                ProbeConfig("av_spam_scanning.GTphish", "antivirus_spam", "GTphish phishing test", SeverityLevel.LOW),
            ],
            "test": [
                ProbeConfig("test.Test", "test", "Basic functionality test", SeverityLevel.INFO),
            ],
        }

    def get_probes_by_category(self, categories: List[ScanCategory]) -> List[str]:
        """Get probe names filtered by categories"""
        selected: List[str] = []

        for category in categories:
            if category == ScanCategory.COMPREHENSIVE:
                for cat_name, probes in self.probes.items():
                    if cat_name != "test":
                        selected.extend([p.name for p in probes if p.enabled])

            elif category == ScanCategory.JAILBREAKS:
                selected.extend([p.name for p in self.probes["jailbreaks"] if p.enabled])

            elif category == ScanCategory.PROMPT_INJECTION:
                selected.extend([p.name for p in self.probes["prompt_injection"] if p.enabled])

            elif category == ScanCategory.DATA_LEAKAGE:
                selected.extend([p.name for p in self.probes["data_leakage"] if p.enabled])

            elif category == ScanCategory.TOXICITY:
                selected.extend([p.name for p in self.probes["toxicity"] if p.enabled])

            elif category == ScanCategory.MALWARE:
                selected.extend([p.name for p in self.probes["malware"] if p.enabled])

            elif category == ScanCategory.HALLUCINATION:
                selected.extend([p.name for p in self.probes["hallucination"] if p.enabled])

            elif category == ScanCategory.ENCODING:
                selected.extend([p.name for p in self.probes["encoding"] if p.enabled])

            elif category == ScanCategory.TEST:
                selected.extend([p.name for p in self.probes["test"] if p.enabled])

        # Remove duplicates while preserving order-ish
        return list(dict.fromkeys(selected))


# ----------------------------
# Scanner
# ----------------------------

class GarakScanner:
    """Advanced Garak scanner wrapper that shells out to garak via -m"""

    def __init__(self, config: ScanConfig):
        self.config = config
        self.probe_registry = GarakProbeRegistry()

        # API keys
        if config.api_keys:
            for k, v in config.api_keys.items():
                if v:
                    os.environ[k] = v

        # Normalize & ensure output directory exists
        out = Path(config.output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        self.config.output_dir = str(out)

        # Add a file logger in output_dir if possible
        self._ensure_file_logging(out)

    def _ensure_file_logging(self, output_dir: Path) -> None:
        log_path = output_dir / "garak_scanner.log"
        # Avoid duplicate file handlers
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                return
        try:
            fh = logging.FileHandler(str(log_path))
            fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            logger.addHandler(fh)
        except Exception:
            # If filesystem is read-only or restricted, keep stdout logging only.
            logger.warning("Could not create file log handler; continuing with stdout logging only.")

    @staticmethod
    def _py_executable() -> str:
        """Return the running interpreter path (Streamlit Cloud-safe)."""
        return sys.executable or "python"

    def _validate_environment(self) -> bool:
        """Validate garak is importable in the *current* environment."""
        try:
            result = subprocess.run(
                [self._py_executable(), "-m", "garak", "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                env=os.environ.copy(),
            )
            if result.returncode == 0:
                logger.info("Garak installation validated successfully (venv interpreter)")
                return True
            logger.error("Garak validation failed: %s", (result.stderr or result.stdout).strip())
            return False
        except subprocess.TimeoutExpired:
            logger.error("Garak validation timed out")
            return False
        except FileNotFoundError as e:
            logger.error("Interpreter not found for garak validation: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error validating garak: %s", e)
            return False

    def _build_garak_command(self, probes: List[str]) -> List[str]:
        """Build the garak command using sys.executable."""
        report_prefix_path = str(Path(self.config.output_dir) / self.config.report_prefix)

        cmd = [
            self._py_executable(),
            "-m",
            "garak",
            "--model_type",
            self.config.model_type,
            "--model_name",
            self.config.target_model,
            "--report_prefix",
            report_prefix_path,
            "--generations",
            str(self.config.max_generations),
        ]

        if probes:
            cmd.extend(["--probes", ",".join(probes)])

        if self.config.exclude_probes:
            cmd.extend(["--probe_exclude", ",".join(self.config.exclude_probes)])

        return cmd

    def run_scan_batch(self, probe_batch: List[str], batch_name: str) -> Dict:
        """Run a batch of probes and return results."""
        logger.info("Starting scan batch: %s", batch_name)
        logger.info("Probes in batch: %s", ", ".join(probe_batch))

        cmd = self._build_garak_command(probe_batch)

        try:
            start = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                cwd=str(Path(self.config.output_dir)),  # keep outputs/logs contained
                env=os.environ.copy(),
            )
            duration = time.time() - start

            batch_result = {
                "batch_name": batch_name,
                "probes": probe_batch,
                "duration": duration,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0,
                "timestamp": datetime.now().isoformat(),
                "command": cmd,  # helpful for debugging
            }

            if batch_result["success"]:
                logger.info("Batch %s completed successfully in %.2fs", batch_name, duration)
            else:
                logger.error("Batch %s failed (rc=%s)", batch_name, result.returncode)
                if result.stderr:
                    logger.error("stderr: %s", result.stderr[:2000])

            return batch_result

        except subprocess.TimeoutExpired:
            logger.error("Batch %s timed out after %ss", batch_name, self.config.timeout)
            return {
                "batch_name": batch_name,
                "probes": probe_batch,
                "duration": self.config.timeout,
                "return_code": -1,
                "stdout": "",
                "stderr": "Scan timed out",
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "command": cmd,
            }
        except Exception as e:
            logger.error("Unexpected error in batch %s: %s", batch_name, e)
            return {
                "batch_name": batch_name,
                "probes": probe_batch,
                "duration": 0,
                "return_code": -1,
                "stdout": "",
                "stderr": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat(),
                "command": cmd,
            }

    def run_comprehensive_scan(self) -> Dict:
        """Run a comprehensive security scan with all selected probe categories."""
        if not self._validate_environment():
            raise RuntimeError("Garak environment validation failed (is garak installed in this venv?)")

        logger.info("Starting LLM security scan")
        logger.info("Target: %s:%s", self.config.model_type, self.config.target_model)
        logger.info("Categories: %s", [c.value for c in self.config.scan_categories])

        selected_probes = self.probe_registry.get_probes_by_category(self.config.scan_categories)

        if self.config.custom_probes:
            selected_probes.extend(self.config.custom_probes)

        if self.config.exclude_probes:
            selected_probes = [p for p in selected_probes if p not in self.config.exclude_probes]

        selected_probes = list(dict.fromkeys(selected_probes))
        logger.info("Total probes to run: %s", len(selected_probes))

        # Batch splitting
        parallel = max(1, int(self.config.parallel_probes))
        batch_size = max(1, (len(selected_probes) + parallel - 1) // parallel)
        probe_batches = [selected_probes[i : i + batch_size] for i in range(0, len(selected_probes), batch_size)]

        # Serialize config
        config_dict = asdict(self.config)
        config_dict["scan_categories"] = [c.value for c in self.config.scan_categories]

        scan_results = {
            "scan_config": config_dict,
            "start_time": datetime.now().isoformat(),
            "total_probes": len(selected_probes),
            "batches": [],
            "summary": {},
        }

        for i, batch in enumerate(probe_batches):
            batch_name = f"batch_{i+1:03d}"
            scan_results["batches"].append(self.run_scan_batch(batch, batch_name))

        scan_results["end_time"] = datetime.now().isoformat()
        scan_results["total_duration"] = sum(b["duration"] for b in scan_results["batches"])
        scan_results["successful_batches"] = sum(1 for b in scan_results["batches"] if b["success"])
        scan_results["failed_batches"] = len(scan_results["batches"]) - scan_results["successful_batches"]
        scan_results["summary"] = self._generate_summary(scan_results)

        self._save_results(scan_results)
        return scan_results

    @staticmethod
    def _generate_summary(scan_results: Dict) -> Dict:
        batches = scan_results.get("batches", [])
        total = len(batches) or 1
        total_dur = scan_results.get("total_duration", 0.0)
        return {
            "total_batches": len(batches),
            "successful_batches": scan_results.get("successful_batches", 0),
            "failed_batches": scan_results.get("failed_batches", 0),
            "success_rate": (scan_results.get("successful_batches", 0) / total),
            "total_runtime": total_dur,
            "average_batch_time": (total_dur / total),
        }

    def _save_results(self, results: Dict) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path(self.config.output_dir)

        json_path = outdir / f"scan_results_{timestamp}.json"
        json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info("Scan results saved to: %s", json_path)

        report_path = outdir / f"scan_report_{timestamp}.md"
        self._generate_markdown_report(results, str(report_path))
        logger.info("Scan report saved to: %s", report_path)

    def _generate_markdown_report(self, results: Dict, output_path: str) -> None:
        report = []
        report.append("# LLM Security Scan Report\n")
        report.append("## Scan Configuration\n")
        report.append(f"- **Target Model**: {results['scan_config']['model_type']}:{results['scan_config']['target_model']}\n")
        report.append(f"- **Scan Categories**: {', '.join(results['scan_config']['scan_categories'])}\n")
        report.append(f"- **Start Time**: {results['start_time']}\n")
        report.append(f"- **End Time**: {results['end_time']}\n")
        report.append(f"- **Total Duration**: {results['total_duration']:.2f} seconds\n\n")

        report.append("## Summary\n")
        report.append(f"- **Total Probes**: {results['total_probes']}\n")
        report.append(f"- **Total Batches**: {results['summary']['total_batches']}\n")
        report.append(f"- **Successful Batches**: {results['summary']['successful_batches']}\n")
        report.append(f"- **Failed Batches**: {results['summary']['failed_batches']}\n")
        report.append(f"- **Success Rate**: {results['summary']['success_rate']:.1%}\n")
        report.append(f"- **Average Batch Time**: {results['summary']['average_batch_time']:.2f}s\n\n")

        report.append("## Batch Results\n\n")
        for batch in results["batches"]:
            status = "✅ SUCCESS" if batch["success"] else "❌ FAILED"
            report.append(f"### {batch['batch_name']} - {status}\n")
            report.append(f"- **Duration**: {batch['duration']:.2f}s\n")
            report.append(f"- **Return Code**: {batch['return_code']}\n")
            report.append(f"- **Probes**: {', '.join(batch['probes'])}\n\n")
            if not batch["success"] and batch.get("stderr"):
                stderr = batch["stderr"]
                if len(stderr) > 800:
                    stderr = stderr[:800] + "..."
                report.append("**Error Output:**\n")
                report.append("```\n" + stderr + "\n```\n\n")

        report.append("## Recommendations\n\n")
        report.append("1. **Review Failed Batches**: Investigate probe batches that failed.\n")
        report.append("2. **Analyze Garak Reports**: Review the JSONL/HTML reports Garak generates.\n")
        report.append("3. **Prioritize Findings**: Address CRITICAL and HIGH findings first.\n")
        report.append("4. **Retest**: Re-run relevant categories after fixes.\n\n")
        report.append("---\n")
        report.append("*Report generated by Garak LLM Security Scanner Toolkit*\n")

        Path(output_path).write_text("".join(report), encoding="utf-8")


# ----------------------------
# Config helpers
# ----------------------------

def create_config_from_args(args: argparse.Namespace) -> ScanConfig:
    scan_categories: List[ScanCategory] = []
    if args.category:
        for cat in args.category:
            try:
                scan_categories.append(ScanCategory(cat))
            except ValueError:
                logger.warning("Unknown category: %s", cat)

    if not scan_categories:
        scan_categories = [ScanCategory.COMPREHENSIVE]

    api_keys: Dict[str, str] = {}
    if args.openai_key:
        api_keys["OPENAI_API_KEY"] = args.openai_key
    if args.anthropic_key:
        api_keys["ANTHROPIC_API_KEY"] = args.anthropic_key
    if args.perspective_key:
        api_keys["PERSPECTIVE_API_KEY"] = args.perspective_key
    if args.cohere_key:
        api_keys["COHERE_API_KEY"] = args.cohere_key

    return ScanConfig(
        target_model=args.model_name,
        model_type=args.model_type,
        scan_categories=scan_categories,
        output_dir=args.output_dir,
        report_prefix=args.report_prefix,
        max_generations=args.max_generations,
        timeout=args.timeout,
        parallel_probes=args.parallel_probes,
        custom_probes=args.custom_probes.split(",") if args.custom_probes else None,
        exclude_probes=args.exclude_probes.split(",") if args.exclude_probes else None,
        api_keys=api_keys if api_keys else None,
    )


def load_config_from_file(config_path: str) -> ScanConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}

    categories: List[ScanCategory] = []
    for cat in config_data.get("scan_categories", ["comprehensive"]):
        try:
            categories.append(ScanCategory(cat))
        except ValueError:
            logger.warning("Unknown category in config: %s", cat)

    return ScanConfig(
        target_model=config_data["target_model"],
        model_type=config_data["model_type"],
        scan_categories=categories or [ScanCategory.COMPREHENSIVE],
        output_dir=config_data.get("output_dir", "./garak_results"),
        report_prefix=config_data.get("report_prefix", "scan"),
        max_generations=int(config_data.get("max_generations", 10)),
        timeout=int(config_data.get("timeout", 3600)),
        parallel_probes=int(config_data.get("parallel_probes", 1)),
        custom_probes=config_data.get("custom_probes"),
        exclude_probes=config_data.get("exclude_probes"),
        api_keys=config_data.get("api_keys"),
    )


def generate_sample_config() -> None:
    sample_config = {
        "target_model": "gpt-3.5-turbo",
        "model_type": "openai",
        "scan_categories": ["jailbreaks", "prompt_injection", "toxicity"],
        "output_dir": "./garak_results",
        "report_prefix": "security_scan",
        "max_generations": 10,
        "timeout": 3600,
        "parallel_probes": 2,
        "custom_probes": None,
        "exclude_probes": ["test.Test"],
        "api_keys": {
            "OPENAI_API_KEY": "your_openai_key_here",
            "PERSPECTIVE_API_KEY": "your_perspective_key_here",
        },
    }

    Path("garak_config_sample.yaml").write_text(
        yaml.dump(sample_config, default_flow_style=False, indent=2),
        encoding="utf-8",
    )
    print("Sample configuration file generated: garak_config_sample.yaml")


# ----------------------------
# Analyzer (kept as in your original)
# ----------------------------

class GarakAnalyzer:
    """Analyzer for Garak JSONL reports"""

    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.probe_registry = GarakProbeRegistry()

    def analyze_jsonl_reports(self) -> Dict:
        jsonl_files = list(self.results_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.warning("No JSONL report files found")
            return {}

        analysis = {
            "total_files": len(jsonl_files),
            "vulnerability_summary": {},
            "probe_results": {},
            "severity_breakdown": {level.value: 0 for level in SeverityLevel},
            "recommendations": [],
        }

        total_vulnerabilities = 0

        for jsonl_file in jsonl_files:
            file_analysis = self._analyze_single_jsonl(jsonl_file)

            for probe, results in file_analysis.get("probe_results", {}).items():
                if probe not in analysis["probe_results"]:
                    analysis["probe_results"][probe] = {
                        "total_attempts": 0,
                        "failed_attempts": 0,
                        "success_rate": 0.0,
                        "vulnerabilities": [],
                    }
                probe_data = analysis["probe_results"][probe]
                probe_data["total_attempts"] += results["total_attempts"]
                probe_data["failed_attempts"] += results["failed_attempts"]
                probe_data["vulnerabilities"].extend(results["vulnerabilities"])

            total_vulnerabilities += file_analysis.get("total_vulnerabilities", 0)

        for probe, data in analysis["probe_results"].items():
            if data["total_attempts"] > 0:
                data["success_rate"] = (data["failed_attempts"] / data["total_attempts"]) * 100.0

            probe_config = self._find_probe_config(probe)
            if probe_config and data["failed_attempts"] > 0:
                analysis["severity_breakdown"][probe_config.severity.value] += data["failed_attempts"]

        analysis["total_vulnerabilities"] = total_vulnerabilities
        analysis["recommendations"] = self._generate_recommendations(analysis)
        return analysis

    def _analyze_single_jsonl(self, jsonl_path: Path) -> Dict:
        probe_results: Dict[str, Dict] = {}
        total_vulnerabilities = 0

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as e:
                        logger.warning("Failed to parse JSON line in %s: %s", jsonl_path, e)
                        continue

                    probe_name = record.get("probe", "unknown")
                    status = record.get("status", "unknown")

                    probe_results.setdefault(
                        probe_name,
                        {"total_attempts": 0, "failed_attempts": 0, "vulnerabilities": []},
                    )
                    probe_results[probe_name]["total_attempts"] += 1

                    if status == "FAIL" or record.get("score", 0) > 0:
                        probe_results[probe_name]["failed_attempts"] += 1
                        probe_results[probe_name]["vulnerabilities"].append(
                            {
                                "prompt": record.get("prompt", ""),
                                "output": record.get("output", ""),
                                "score": record.get("score", 0),
                                "detector": record.get("detector", ""),
                            }
                        )
                        total_vulnerabilities += 1

        except Exception as e:
            logger.error("Error analyzing %s: %s", jsonl_path, e)

        return {
            "file": str(jsonl_path),
            "probe_results": probe_results,
            "total_vulnerabilities": total_vulnerabilities,
        }

    def _find_probe_config(self, probe_name: str) -> Optional[ProbeConfig]:
        for category_probes in self.probe_registry.probes.values():
            for probe_config in category_probes:
                if probe_config.name == probe_name:
                    return probe_config
        return None

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        recommendations: List[str] = []
        total_vulns = analysis.get("total_vulnerabilities", 0)

        if total_vulns == 0:
            return ["✅ No vulnerabilities detected in the current scan"]

        sev = analysis.get("severity_breakdown", {})
        if sev.get("critical", 0) > 0:
            recommendations.append(f"🚨 CRITICAL: {sev['critical']} critical vulnerabilities found - immediate action required")
        if sev.get("high", 0) > 0:
            recommendations.append(f"⚠️ HIGH: {sev['high']} high-severity vulnerabilities - address within 24-48 hours")
        if sev.get("medium", 0) > 0:
            recommendations.append(f"🔶 MEDIUM: {sev['medium']} medium-severity vulnerabilities - address within 1 week")

        vulnerable_probes = {k: v for k, v in analysis.get("probe_results", {}).items() if v.get("failed_attempts", 0) > 0}

        if any("dan." in probe for probe in vulnerable_probes):
            recommendations.append("🔒 Implement stronger jailbreak defenses - DAN vulnerabilities detected")
        if any("malwaregen." in probe for probe in vulnerable_probes):
            recommendations.append("🛡️ Add malware generation filters - model generates potentially harmful code")
        if any("promptinject." in probe for probe in vulnerable_probes):
            recommendations.append("🎯 Strengthen prompt injection defenses - injection attacks successful")
        if any("encoding." in probe for probe in vulnerable_probes):
            recommendations.append("🔤 Implement encoding-aware input validation - bypass techniques working")
        if any("leakreplay." in probe for probe in vulnerable_probes):
            recommendations.append("🔐 Review training data exposure - potential data leakage detected")

        recommendations.append("📊 Run follow-up scans after implementing fixes to verify improvements")
        return recommendations

    def generate_detailed_report(self, output_path: str):
        analysis = self.analyze_jsonl_reports()
        if not analysis:
            Path(output_path).write_text("# Detailed Garak Security Analysis Report\n\nNo JSONL files found.\n", encoding="utf-8")
            logger.info("Detailed analysis report saved to: %s", output_path)
            return

        report_content = [
            "# Detailed Garak Security Analysis Report\n\n",
            "## Executive Summary\n",
            f"- **Total Vulnerabilities Found**: {analysis.get('total_vulnerabilities', 0)}\n",
            f"- **Files Analyzed**: {analysis.get('total_files', 0)}\n",
            f"- **Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "## Severity Breakdown\n",
        ]

        for severity, count in analysis.get("severity_breakdown", {}).items():
            if count > 0:
                emoji = {"critical": "🚨", "high": "⚠️", "medium": "🔶", "low": "🔵", "info": "ℹ️"}.get(severity, "•")
                report_content.append(f"- **{severity.upper()}**: {count} {emoji}\n")

        report_content.append("\n## Recommendations\n\n")
        for i, rec in enumerate(analysis.get("recommendations", []), 1):
            report_content.append(f"{i}. {rec}\n")

        Path(output_path).write_text("".join(report_content), encoding="utf-8")
        logger.info("Detailed analysis report saved to: %s", output_path)


# ----------------------------
# CLI
# ----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive Garak LLM Security Scanner Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python garak_scanner_toolkit.py --model-type openai --model-name gpt-3.5-turbo
  python garak_scanner_toolkit.py --model-type huggingface --model-name gpt2 --category comprehensive
  python garak_scanner_toolkit.py --model-type openai --model-name gpt-4 --category jailbreaks toxicity
  python garak_scanner_toolkit.py --config config.yaml
  python garak_scanner_toolkit.py --generate-config
  python garak_scanner_toolkit.py --analyze ./results
""",
    )

    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument("--config", type=str, help="Load configuration from YAML file")
    config_group.add_argument("--generate-config", action="store_true", help="Generate sample configuration file")

    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model-type",
        type=str,
        choices=["openai", "huggingface", "anthropic", "cohere", "replicate", "ollama", "test"],
        help="Type of model to scan",
    )
    model_group.add_argument("--model-name", type=str, help="Specific model name to scan")

    scan_group = parser.add_argument_group("Scan Configuration")
    scan_group.add_argument("--category", type=str, nargs="+", choices=[c.value for c in ScanCategory], help="Categories to test")
    scan_group.add_argument("--custom-probes", type=str, help="Comma-separated list of custom probes")
    scan_group.add_argument("--exclude-probes", type=str, help="Comma-separated list of probes to exclude")
    scan_group.add_argument("--max-generations", type=int, default=10, help="Maximum generations per probe")
    scan_group.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds for each batch")
    scan_group.add_argument("--parallel-probes", type=int, default=1, help="Number of parallel probe batches")

    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output-dir", type=str, default="./garak_results", help="Output directory for results")
    output_group.add_argument("--report-prefix", type=str, default="scan", help="Prefix for report files")

    api_group = parser.add_argument_group("API Keys")
    api_group.add_argument("--openai-key", type=str, help="OpenAI API key")
    api_group.add_argument("--anthropic-key", type=str, help="Anthropic API key")
    api_group.add_argument("--perspective-key", type=str, help="Perspective API key")
    api_group.add_argument("--cohere-key", type=str, help="Cohere API key")

    analysis_group = parser.add_argument_group("Analysis")
    analysis_group.add_argument("--analyze", type=str, help="Analyze existing results in specified directory")

    utility_group = parser.add_argument_group("Utilities")
    utility_group.add_argument("--list-probes", action="store_true", help="List all available probes by category")
    utility_group.add_argument("--validate", action="store_true", help="Validate Garak installation")

    args = parser.parse_args()

    if args.generate_config:
        generate_sample_config()
        return

    if args.list_probes:
        registry = GarakProbeRegistry()
        print("\n=== Available Garak Probes by Category ===\n")
        for category, probes in registry.probes.items():
            print(f"📁 {category.upper()}:")
            for probe in probes:
                status = "✅" if probe.enabled else "❌"
                print(f"  {status} {probe.name} - {probe.description} [{probe.severity.value}]")
            print()
        return

    if args.validate:
        scanner = GarakScanner(
            ScanConfig(
                target_model="test",
                model_type="test",
                scan_categories=[ScanCategory.TEST],
                output_dir="./test",
                report_prefix="test",
            )
        )
        ok = scanner._validate_environment()
        print("✅ Garak installation is valid and ready to use" if ok else "❌ Garak installation validation failed")
        return

    if args.analyze:
        analyzer = GarakAnalyzer(args.analyze)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"analysis_report_{ts}.md"
        analyzer.generate_detailed_report(report_path)
        print(f"Analysis complete. Report saved to: {report_path}")
        return

    # Main scanning
    try:
        if args.config:
            config = load_config_from_file(args.config)
        else:
            if not args.model_type or not args.model_name:
                parser.error("--model-type and --model-name are required unless using --config")
            config = create_config_from_args(args)

        scanner = GarakScanner(config)
        results = scanner.run_comprehensive_scan()

        print("\n=== Scan Complete ===")
        print(f"Total Duration: {results['total_duration']:.2f}s")
        print(f"Successful Batches: {results['successful_batches']}/{len(results['batches'])}")
        print(f"Results saved to: {config.output_dir}")

        if results["successful_batches"] > 0:
            print("\n=== Running Analysis ===")
            analyzer = GarakAnalyzer(config.output_dir)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_path = str(Path(config.output_dir) / f"analysis_report_{ts}.md")
            analyzer.generate_detailed_report(analysis_path)
            print(f"Analysis report saved to: {analysis_path}")

    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
    except Exception as e:
        logger.error("Scan failed: %s", e)
        raise


if __name__ == "__main__":
    main()
