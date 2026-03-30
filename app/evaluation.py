"""
RAG Evaluation System using RAGAS for AutoInsight AI
Provides comprehensive evaluation metrics for RAG system performance
"""

try:
    import ragas
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: RAGAS not available. Evaluation features will be disabled.")

# Import LLM related modules separately
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("Warning: LLM modules not available for RAGAS evaluation.")

import pandas as pd
from typing import List, Dict, Any, Optional
import json
import os
from datetime import datetime


class RAGEvaluator:
    """RAG system evaluator using RAGAS metrics"""

    def __init__(self):
        self.use_ragas = RAGAS_AVAILABLE and LLM_AVAILABLE
        if not self.use_ragas:
            print("RAGAS evaluation not available - using basic evaluation")
            self.metrics_without_gt = []
            self.metrics_with_gt = []
            self.llm = None
        else:
            # Use Gemini for evaluation
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.1,  # Lower temperature for evaluation consistency
                api_key=os.getenv("GOOGLE_API_KEY")
            )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                api_key=os.getenv("GOOGLE_API_KEY")
            )
            # Metrics that don't require ground truth
            self.metrics_without_gt = [
                faithfulness,
                answer_relevancy
            ]
            # Metrics that require ground truth
            self.metrics_with_gt = [
                context_precision,
                context_recall,
                answer_similarity
            ]
        self.evaluation_history = []

    def evaluate_qa_pair(self,
                         question: str,
                         answer: str,
                         contexts: List[str],
                         ground_truth: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a single Q&A pair

        Args:
            question: The question asked
            answer: The generated answer
            contexts: List of retrieved context chunks
            ground_truth: Optional ground truth answer for comparison

        Returns:
            Dictionary with evaluation scores
        """
        if not self.use_ragas:
            # Basic evaluation without RAGAS
            return {
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'answer': answer,
                'contexts': contexts,
                'ground_truth': ground_truth,
                'scores': {
                    'basic_relevance': self._basic_relevance_score(question, answer),
                    'basic_context_coverage': self._basic_context_coverage(answer, contexts)
                },
                'overall_score': (self._basic_relevance_score(question, answer) +
                                self._basic_context_coverage(answer, contexts)) / 2
            }

        # Choose metrics based on whether ground truth is provided
        if ground_truth is not None:
            metrics_to_use = self.metrics_without_gt + self.metrics_with_gt
        else:
            metrics_to_use = self.metrics_without_gt

        # Prepare data for RAGAS
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [contexts],
        }

        if ground_truth:
            data['reference'] = [ground_truth]  # RAGAS uses 'reference' for ground truth

        dataset = Dataset.from_dict(data)

        # Run evaluation
        if self.llm is not None:
            results = evaluate(dataset, metrics_to_use, llm=self.llm, embeddings=self.embeddings)
        else:
            results = evaluate(dataset, metrics_to_use)

        # Convert to dictionary
        scores = {}
        for metric in metrics_to_use:
            metric_name = getattr(metric, 'name', metric.__class__.__name__.lower())
            scores[metric_name] = float(results[metric_name][0])

        # Add metadata
        evaluation_result = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'scores': scores,
            'overall_score': sum(scores.values()) / len(scores)
        }

        self.evaluation_history.append(evaluation_result)
        return evaluation_result

    def _basic_relevance_score(self, question: str, answer: str) -> float:
        """Basic relevance scoring without RAGAS"""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        overlap = len(question_words.intersection(answer_words))
        return min(1.0, overlap / max(len(question_words), 1))

    def _basic_context_coverage(self, answer: str, contexts: List[str]) -> float:
        """Basic context coverage scoring without RAGAS"""
        context_text = ' '.join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())

        overlap = len(answer_words.intersection(context_words))
        return min(1.0, overlap / max(len(answer_words), 1))

    def evaluate_batch(self,
                       questions: List[str],
                       answers: List[str],
                       contexts_list: List[List[str]],
                       ground_truths: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a batch of Q&A pairs

        Args:
            questions: List of questions
            answers: List of generated answers
            contexts_list: List of context lists for each question
            ground_truths: Optional list of ground truth answers

        Returns:
            Dictionary with batch evaluation results
        """
        if not RAGAS_AVAILABLE:
            # Basic batch evaluation
            individual_results = []
            for i, (q, a, c) in enumerate(zip(questions, answers, contexts_list)):
                result = self.evaluate_qa_pair(q, a, c, ground_truths[i] if ground_truths else None)
                individual_results.append(result)

            avg_relevance = sum(r['scores']['basic_relevance'] for r in individual_results) / len(individual_results)
            avg_coverage = sum(r['scores']['basic_context_coverage'] for r in individual_results) / len(individual_results)

            return {
                'timestamp': datetime.now().isoformat(),
                'batch_size': len(questions),
                'average_scores': {
                    'avg_basic_relevance': avg_relevance,
                    'avg_basic_context_coverage': avg_coverage
                },
                'overall_average': (avg_relevance + avg_coverage) / 2,
                'individual_results': individual_results
            }

        # Full RAGAS batch evaluation
        # Prepare data
        data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts_list,
        }

        if ground_truths:
            data['ground_truth'] = ground_truths

        dataset = Dataset.from_dict(data)

        # Run evaluation
        results = evaluate(dataset, self.metrics)

        # Calculate averages
        avg_scores = {}
        for metric in self.metrics:
            metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
            avg_scores[f'avg_{metric_name}'] = float(results[metric_name].mean())

        batch_result = {
            'timestamp': datetime.now().isoformat(),
            'batch_size': len(questions),
            'average_scores': avg_scores,
            'overall_average': sum(avg_scores.values()) / len(avg_scores),
            'individual_results': []
        }

        # Add individual results
        for i in range(len(questions)):
            individual_scores = {}
            for metric in self.metrics:
                metric_name = metric.__name__ if hasattr(metric, '__name__') else str(metric)
                individual_scores[metric_name] = float(results[metric_name][i])

            individual_result = {
                'question': questions[i],
                'answer': answers[i],
                'contexts': contexts_list[i],
                'ground_truth': ground_truths[i] if ground_truths else None,
                'scores': individual_scores,
                'overall_score': sum(individual_scores.values()) / len(individual_scores)
            }
            batch_result['individual_results'].append(individual_result)
            self.evaluation_history.append(individual_result)

        return batch_result

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of all evaluations"""
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}

        total_evaluations = len(self.evaluation_history)
        scores_by_metric = {}

        # Collect all scores
        for result in self.evaluation_history:
            if 'scores' in result:
                for metric, score in result['scores'].items():
                    if metric not in scores_by_metric:
                        scores_by_metric[metric] = []
                    scores_by_metric[metric].append(score)

        # Calculate averages
        avg_scores = {}
        for metric, scores in scores_by_metric.items():
            avg_scores[f'avg_{metric}'] = sum(scores) / len(scores)

        return {
            'total_evaluations': total_evaluations,
            'average_scores': avg_scores,
            'overall_average': sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0,
            'score_distributions': scores_by_metric
        }

    def save_evaluation_history(self, filepath: str = 'evaluation_history.json'):
        """Save evaluation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2, default=str)

    def load_evaluation_history(self, filepath: str = 'evaluation_history.json'):
        """Load evaluation history from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.evaluation_history = json.load(f)

    def generate_evaluation_report(self) -> str:
        """Generate a human-readable evaluation report"""
        summary = self.get_evaluation_summary()

        if 'message' in summary:
            return summary['message']

        report = f"""# RAG Evaluation Report

## Summary
- Total Evaluations: {summary['total_evaluations']}
- Overall Average Score: {summary['overall_average']:.3f}

## Average Scores by Metric
"""

        for metric, score in summary['average_scores'].items():
            report += f"- {metric}: {score:.3f}\n"

        report += "\n## Recommendations\n"

        # Add recommendations based on scores
        if RAGAS_AVAILABLE:
            avg_faithfulness = summary['average_scores'].get('avg_faithfulness', 0)
            avg_relevancy = summary['average_scores'].get('avg_answer_relevancy', 0)
            avg_context_relevancy = summary['average_scores'].get('avg_context_precision', 0)

            if avg_faithfulness < 0.7:
                report += "- Consider improving answer faithfulness - answers may not be well-grounded in context\n"
            if avg_relevancy < 0.7:
                report += "- Work on answer relevance - responses may not adequately address the questions\n"
            if avg_context_relevancy < 0.7:
                report += "- Improve context retrieval - retrieved information may not be relevant to questions\n"

            if all(score > 0.8 for score in summary['average_scores'].values()):
                report += "- Excellent performance! All metrics are above 0.8\n"
        else:
            report += "- Using basic evaluation metrics (RAGAS not available)\n"
            report += "- Consider installing RAGAS for comprehensive evaluation: pip install ragas\n"

        return report


# Global evaluator instance
evaluator = RAGEvaluator()


def evaluate_rag_response(question: str, answer: str, contexts: List[str],
                         ground_truth: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single RAG response

    Args:
        question: The question asked
        answer: The generated answer
        contexts: Retrieved context chunks
        ground_truth: Optional ground truth answer

    Returns:
        Evaluation results
    """
    return evaluator.evaluate_qa_pair(question, answer, contexts, ground_truth)


def get_evaluation_report() -> str:
    """Get a formatted evaluation report"""
    return evaluator.generate_evaluation_report()