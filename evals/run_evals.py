"""Evaluation framework for RAG system quality metrics."""

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

from src.rag_system.pipeline import create_pipeline
from src.rag_system.utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


class RAGEvaluator:
    """Evaluates RAG system performance and quality."""

    def __init__(self):
        """Initialize evaluator."""
        self.results: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "tests": [],
        }

    async def evaluate_config_validation(self) -> Dict[str, Any]:
        """Evaluate configuration validation."""
        logger.info("Running config validation test")
        test_result = {
            "name": "configuration_validation",
            "status": "pass",
            "details": {},
        }

        try:
            from src.rag_system.config import get_config

            config = get_config()
            test_result["details"]["environment"] = config.environment
            test_result["details"]["debug_mode"] = config.is_debug
            test_result["details"]["is_production"] = config.is_production

        except Exception as e:
            test_result["status"] = "fail"
            test_result["details"]["error"] = str(e)
            logger.error(f"Config validation failed: {str(e)}")

        return test_result

    async def evaluate_component_initialization(self) -> Dict[str, Any]:
        """Evaluate component initialization."""
        logger.info("Running component initialization test")
        test_result = {
            "name": "component_initialization",
            "status": "pass",
            "details": {
                "components": {},
            },
        }

        try:
            from src.rag_system.components import PDFParser, VisionProcessor, VectorIndexer

            # Test PDF Parser
            try:
                parser = PDFParser()
                test_result["details"]["components"]["pdf_parser"] = "initialized"
            except Exception as e:
                test_result["status"] = "fail"
                test_result["details"]["components"]["pdf_parser"] = f"failed: {str(e)}"

            # Test Vision Processor
            try:
                processor = VisionProcessor()
                test_result["details"]["components"]["vision_processor"] = "initialized"
            except Exception as e:
                test_result["status"] = "fail"
                test_result["details"]["components"]["vision_processor"] = f"failed: {str(e)}"

            # Test Vector Indexer
            try:
                indexer = VectorIndexer()
                test_result["details"]["components"]["vector_indexer"] = "initialized"
            except Exception as e:
                test_result["status"] = "fail"
                test_result["details"]["components"]["vector_indexer"] = f"failed: {str(e)}"

        except Exception as e:
            test_result["status"] = "fail"
            test_result["details"]["error"] = str(e)
            logger.error(f"Component initialization test failed: {str(e)}")

        return test_result

    async def evaluate_logging(self) -> Dict[str, Any]:
        """Evaluate logging infrastructure."""
        logger.info("Running logging infrastructure test")
        test_result = {
            "name": "logging_infrastructure",
            "status": "pass",
            "details": {},
        }

        try:
            from src.rag_system.utils.logger import get_logger

            test_logger = get_logger("test_logger")
            test_logger.info("Test log message", test_key="test_value")
            test_result["details"]["logger_created"] = True

        except Exception as e:
            test_result["status"] = "fail"
            test_result["details"]["error"] = str(e)

        return test_result

    async def evaluate_exceptions(self) -> Dict[str, Any]:
        """Evaluate custom exception handling."""
        logger.info("Running exception handling test")
        test_result = {
            "name": "exception_handling",
            "status": "pass",
            "details": {"exceptions_tested": []},
        }

        try:
            from src.rag_system.utils.exceptions import (
                RAGException,
                PDFParsingError,
                VisionParsingError,
                VectorStorageError,
                APIRateLimitError,
            )

            exceptions_to_test = [
                ("RAGException", RAGException("test", code="TEST")),
                ("PDFParsingError", PDFParsingError("test", file_path="test.pdf")),
                ("VisionParsingError", VisionParsingError("test", image_path="test.png")),
                ("VectorStorageError", VectorStorageError("test", dataset_path="hub://test")),
                ("APIRateLimitError", APIRateLimitError("test", retry_after=60)),
            ]

            for exc_name, exc_instance in exceptions_to_test:
                try:
                    raise exc_instance
                except RAGException as e:
                    test_result["details"]["exceptions_tested"].append({
                        "name": exc_name,
                        "code": e.code,
                        "status": "ok",
                    })

        except Exception as e:
            test_result["status"] = "fail"
            test_result["details"]["error"] = str(e)

        return test_result

    async def evaluate_rate_limiting(self) -> Dict[str, Any]:
        """Evaluate rate limiting functionality."""
        logger.info("Running rate limiting test")
        test_result = {
            "name": "rate_limiting",
            "status": "pass",
            "details": {},
        }

        try:
            from src.rag_system.utils.rate_limiter import AsyncRateLimiter

            limiter = AsyncRateLimiter(requests_per_second=10.0, burst_size=5)

            # Test acquiring tokens
            start = time.time()
            await limiter.acquire()
            elapsed = time.time() - start

            test_result["details"]["initial_acquisition_time"] = round(elapsed, 3)
            test_result["details"]["rate_limiter_stats"] = limiter.get_stats()

        except Exception as e:
            test_result["status"] = "fail"
            test_result["details"]["error"] = str(e)

        return test_result

    async def evaluate_retry_policy(self) -> Dict[str, Any]:
        """Evaluate retry policy."""
        logger.info("Running retry policy test")
        test_result = {
            "name": "retry_policy",
            "status": "pass",
            "details": {},
        }

        try:
            from src.rag_system.utils.retry_policy import RetryPolicy

            policy = RetryPolicy(max_attempts=3, base_delay_seconds=0.1)

            # Test delay calculation
            delays = [policy.get_delay(i) for i in range(3)]
            test_result["details"]["calculated_delays"] = [round(d, 3) for d in delays]
            test_result["details"]["delays_increase"] = delays[0] < delays[1] < delays[2]

        except Exception as e:
            test_result["status"] = "fail"
            test_result["details"]["error"] = str(e)

        return test_result

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all evaluation tests."""
        logger.info("Starting comprehensive evaluation suite")

        tests = [
            self.evaluate_config_validation(),
            self.evaluate_component_initialization(),
            self.evaluate_logging(),
            self.evaluate_exceptions(),
            self.evaluate_rate_limiting(),
            self.evaluate_retry_policy(),
        ]

        results = await asyncio.gather(*tests)
        self.results["tests"] = results

        # Calculate summary metrics
        passed = sum(1 for r in results if r["status"] == "pass")
        failed = sum(1 for r in results if r["status"] == "fail")

        self.results["metrics"]["total_tests"] = len(results)
        self.results["metrics"]["passed_tests"] = passed
        self.results["metrics"]["failed_tests"] = failed
        self.results["metrics"]["pass_rate"] = round(passed / len(results) * 100, 2) if results else 0

        return self.results

    def save_report(self, output_path: str = "eval_report.json") -> None:
        """Save evaluation report to file."""
        try:
            with open(output_path, "w") as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Evaluation report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {str(e)}")


async def main():
    """Run evaluation suite."""
    setup_logging(level="INFO", format_type="json")

    evaluator = RAGEvaluator()
    results = await evaluator.run_all_tests()

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Tests: {results['metrics']['total_tests']}")
    print(f"Passed: {results['metrics']['passed_tests']}")
    print(f"Failed: {results['metrics']['failed_tests']}")
    print(f"Pass Rate: {results['metrics']['pass_rate']}%")
    print("=" * 60 + "\n")

    # Save report
    evaluator.save_report("evals/eval_report.json")


if __name__ == "__main__":
    asyncio.run(main())
