"""
Multi-Agent System using CrewAI for AutoInsight AI
Implements specialized agents for document analysis, question answering, and evaluation
"""

try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    print("Warning: CrewAI not available. Multi-agent features will be disabled.")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class MultiAgentRAGSystem:
    """Multi-agent system for enhanced RAG operations"""

    def __init__(self, vectorstore=None, llm=None):
        if not CREWAI_AVAILABLE:
            raise ImportError("CrewAI is required for multi-agent functionality. Please install it with: pip install crewai")

        self.vectorstore = vectorstore
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize agents
        self.document_analyzer = self._create_document_analyzer()
        self.question_answerer = self._create_question_answerer()
        self.evaluator = self._create_evaluator()

    def _create_document_analyzer(self) -> Agent:
        """Create document analysis agent"""
        return Agent(
            role="Document Analysis Specialist",
            goal="Analyze documents and extract key information, themes, and insights",
            backstory="""You are an expert document analyst with years of experience in
            processing technical documents, research papers, and business reports.
            You excel at identifying key concepts, relationships, and actionable insights.""",
            llm="gemini/gemini-2.5-flash",
            verbose=True,
            allow_delegation=False
        )

    def _create_question_answerer(self) -> Agent:
        """Create question answering agent"""
        return Agent(
            role="RAG Question Answerer",
            goal="Provide accurate, contextual answers based on retrieved documents",
            backstory="""You are a specialized AI assistant trained in retrieval-augmented
            generation. You combine document context with your knowledge to provide
            comprehensive, accurate answers. You always prioritize information from
            the provided context.""",
            llm="gemini/gemini-2.5-flash",
            verbose=True,
            allow_delegation=False
        )

    def _create_evaluator(self) -> Agent:
        """Create evaluation agent"""
        return Agent(
            role="RAG System Evaluator",
            goal="Evaluate the quality and accuracy of RAG system responses",
            backstory="""You are a quality assurance specialist for AI systems.
            You evaluate response accuracy, relevance, completeness, and helpfulness.
            You provide detailed feedback on system performance.""",
            llm="gemini/gemini-2.5-flash",
            verbose=True,
            allow_delegation=False
        )

    def analyze_document(self, documents: List[Dict]) -> str:
        """
        Analyze documents using the document analyzer agent

        Args:
            documents: List of document dictionaries with content

        Returns:
            Analysis summary
        """
        analysis_task = Task(
            description=f"""Analyze the following documents and provide a comprehensive summary:

Documents: {documents[:3]}  # Limit for context

Provide:
1. Main topics and themes
2. Key insights and findings
3. Important entities mentioned
4. Overall document quality assessment""",
            agent=self.document_analyzer,
            expected_output="A detailed analysis report with key findings and insights"
        )

        crew = Crew(
            agents=[self.document_analyzer],
            tasks=[analysis_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        return result

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer questions using multi-agent approach

        Args:
            question: User question
            context: Retrieved context from documents

        Returns:
            Comprehensive answer
        """
        qa_task = Task(
            description=f"""Answer the following question using the provided context:

Question: {question}

Context: {context}

Provide a comprehensive, accurate answer based only on the context provided.
If the context doesn't contain enough information, clearly state this.""",
            agent=self.question_answerer,
            expected_output="A clear, comprehensive answer based on the provided context"
        )

        crew = Crew(
            agents=[self.question_answerer],
            tasks=[qa_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        return result

    def evaluate_response(self, question: str, context: str, response: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG response

        Args:
            question: Original question
            context: Retrieved context
            response: Generated response

        Returns:
            Evaluation metrics and feedback
        """
        evaluation_task = Task(
            description=f"""Evaluate the following RAG system response:

Question: {question}
Context: {context}
Response: {response}

Evaluate on:
1. Accuracy (1-10): How accurate is the response?
2. Relevance (1-10): How relevant is the response to the question?
3. Completeness (1-10): How complete is the answer?
4. Groundedness (1-10): How well is the response grounded in the context?
5. Overall quality assessment

Provide specific feedback and suggestions for improvement.""",
            agent=self.evaluator,
            expected_output="A detailed evaluation with scores and constructive feedback"
        )

        crew = Crew(
            agents=[self.evaluator],
            tasks=[evaluation_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()

        # Parse evaluation results
        return {
            'evaluation_report': result,
            'question': question,
            'response': response,
            'context_length': len(context)
        }


def create_multi_agent_crew(question: str, context: str) -> str:
    """
    Convenience function to create and run a multi-agent crew for Q&A

    Args:
        question: User question
        context: Retrieved context

    Returns:
        Multi-agent generated answer
    """
    if not CREWAI_AVAILABLE:
        return f"Multi-agent features are not available. Basic response:\n\n{context[:500]}..."

    system = MultiAgentRAGSystem()

    # Run analysis first
    analysis = system.analyze_document([{"content": context, "type": "retrieved_context"}])

    # Then answer the question
    answer = system.answer_question(question, context)

    # Finally evaluate
    evaluation = system.evaluate_response(question, context, answer)

    return f"""**Document Analysis:**\n{analysis}\n\n**Answer:**\n{answer}\n\n**Evaluation:**\n{evaluation['evaluation_report']}"""


class MultiAgentRAGSystem:
    """Multi-agent system for enhanced RAG operations"""

    def __init__(self, vectorstore=None, llm=None):
        self.vectorstore = vectorstore
        self.llm = llm or ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Initialize agents
        self.document_analyzer = self._create_document_analyzer()
        self.question_answerer = self._create_question_answerer()
        self.evaluator = self._create_evaluator()

    def _create_document_analyzer(self) -> Agent:
        """Create document analysis agent"""
        return Agent(
            role="Document Analysis Specialist",
            goal="Analyze documents and extract key information, themes, and insights",
            backstory="""You are an expert document analyst with years of experience in
            processing technical documents, research papers, and business reports.
            You excel at identifying key concepts, relationships, and actionable insights.""",
            llm="gemini/gemini-2.5-flash",
            verbose=True,
            allow_delegation=False
        )

    def _create_question_answerer(self) -> Agent:
        """Create question answering agent"""
        return Agent(
            role="RAG Question Answerer",
            goal="Provide accurate, contextual answers based on retrieved documents",
            backstory="""You are a specialized AI assistant trained in retrieval-augmented
            generation. You combine document context with your knowledge to provide
            comprehensive, accurate answers. You always prioritize information from
            the provided context.""",
            llm="gemini/gemini-2.5-flash",
            verbose=True,
            allow_delegation=False
        )

    def _create_evaluator(self) -> Agent:
        """Create evaluation agent"""
        return Agent(
            role="RAG System Evaluator",
            goal="Evaluate the quality and accuracy of RAG system responses",
            backstory="""You are a quality assurance specialist for AI systems.
            You evaluate response accuracy, relevance, completeness, and helpfulness.
            You provide detailed feedback on system performance.""",
            llm="gemini/gemini-2.5-flash",
            verbose=True,
            allow_delegation=False
        )

    def analyze_document(self, documents: List[Dict]) -> str:
        """
        Analyze documents using the document analyzer agent

        Args:
            documents: List of document dictionaries with content

        Returns:
            Analysis summary
        """
        analysis_task = Task(
            description=f"""Analyze the following documents and provide a comprehensive summary:

Documents: {documents[:3]}  # Limit for context

Provide:
1. Main topics and themes
2. Key insights and findings
3. Important entities mentioned
4. Overall document quality assessment""",
            agent=self.document_analyzer,
            expected_output="A detailed analysis report with key findings and insights"
        )

        crew = Crew(
            agents=[self.document_analyzer],
            tasks=[analysis_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        return result

    def answer_question(self, question: str, context: str) -> str:
        """
        Answer questions using multi-agent approach

        Args:
            question: User question
            context: Retrieved context from documents

        Returns:
            Comprehensive answer
        """
        qa_task = Task(
            description=f"""Answer the following question using the provided context:

Question: {question}

Context: {context}

Provide a comprehensive, accurate answer based only on the context provided.
If the context doesn't contain enough information, clearly state this.""",
            agent=self.question_answerer,
            expected_output="A clear, comprehensive answer based on the provided context"
        )

        crew = Crew(
            agents=[self.question_answerer],
            tasks=[qa_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()
        return result

    def evaluate_response(self, question: str, context: str, response: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a RAG response

        Args:
            question: Original question
            context: Retrieved context
            response: Generated response

        Returns:
            Evaluation metrics and feedback
        """
        evaluation_task = Task(
            description=f"""Evaluate the following RAG system response:

Question: {question}
Context: {context}
Response: {response}

Evaluate on:
1. Accuracy (1-10): How accurate is the response?
2. Relevance (1-10): How relevant is the response to the question?
3. Completeness (1-10): How complete is the answer?
4. Groundedness (1-10): How well is the response grounded in the context?
5. Overall quality assessment

Provide specific feedback and suggestions for improvement.""",
            agent=self.evaluator,
            expected_output="A detailed evaluation with scores and constructive feedback"
        )

        crew = Crew(
            agents=[self.evaluator],
            tasks=[evaluation_task],
            verbose=True,
            process=Process.sequential
        )

        result = crew.kickoff()

        # Parse evaluation results
        return {
            'evaluation_report': result,
            'question': question,
            'response': response,
            'context_length': len(context)
        }


def create_multi_agent_crew(question: str, context: str) -> str:
    """
    Convenience function to create and run a multi-agent crew for Q&A

    Args:
        question: User question
        context: Retrieved context

    Returns:
        Multi-agent generated answer
    """
    system = MultiAgentRAGSystem()

    # Run analysis first
    analysis = system.analyze_document([{"content": context, "type": "retrieved_context"}])

    # Then answer the question
    answer = system.answer_question(question, context)

    # Finally evaluate
    evaluation = system.evaluate_response(question, context, answer)

    return f"""**Document Analysis:**\n{analysis}\n\n**Answer:**\n{answer}\n\n**Evaluation:**\n{evaluation['evaluation_report']}"""
