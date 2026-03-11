"""
Database layer for Inquiro world model.

Handles all SQLite operations - creating tables, CRUD operations,
and converting between Pydantic models and database rows.
"""

import sqlite3
import json
import threading
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from .models import (
    Finding, Relationship, Hypothesis, Task, ResearchQuestion,
    Source, FindingType, RelationshipType, HypothesisStatus,
    TaskType, TaskStatus, TaskPriority, QuestionStatus
)


class Database:
    """
    SQLite database wrapper for the world model.
    
    Handles:
    - Table creation and schema management
    - CRUD operations for all entity types
    - Conversion between Pydantic models and DB rows
    
    Example:
        db = Database("./data/world_model.db")
        finding_id = db.insert_finding(my_finding)
        findings = db.get_findings_by_cycle(3)
    """
    
    def __init__(self, db_path: str = "./data/world_model.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite file. Created if doesn't exist.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._lock = threading.Lock()
        
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Access columns by name
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        
        self._create_tables()
    
    def _create_tables(self) -> None:
        """Create all tables if they don't exist."""
        
        # Findings table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                finding_type TEXT NOT NULL,
                confidence REAL DEFAULT 0.5,
                cycle INTEGER NOT NULL,
                source_json TEXT NOT NULL,
                tags_json TEXT DEFAULT '[]',
                evidence TEXT,
                statistical_support_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Relationships table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                reasoning TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_id) REFERENCES findings(id) ON DELETE CASCADE,
                FOREIGN KEY (to_id) REFERENCES findings(id) ON DELETE CASCADE,
                UNIQUE(from_id, to_id, relationship_type)
            )
        """)
        
        # Hypotheses table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                status TEXT DEFAULT 'proposed',
                cycle_proposed INTEGER NOT NULL,
                cycle_resolved INTEGER,
                supporting_finding_ids_json TEXT DEFAULT '[]',
                contradicting_finding_ids_json TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tasks table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                description TEXT NOT NULL,
                goal TEXT,
                priority TEXT DEFAULT 'medium',
                status TEXT DEFAULT 'pending',
                cycle INTEGER NOT NULL,
                result_finding_ids_json TEXT DEFAULT '[]',
                error_message TEXT,
                execution_time_seconds REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        # Research questions table - for question-driven research
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS research_questions (
                id TEXT PRIMARY KEY,
                question_text TEXT NOT NULL,
                status TEXT DEFAULT 'unanswered',
                priority TEXT DEFAULT 'medium',
                area_id TEXT,
                area_name TEXT,
                keywords_json TEXT DEFAULT '[]',
                related_finding_ids_json TEXT DEFAULT '[]',
                answer_summary TEXT,
                confidence_score REAL DEFAULT 0.0,
                evidence_count INTEGER DEFAULT 0,
                cycle_created INTEGER DEFAULT 1,
                cycle_answered INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                answered_at TIMESTAMP
            )
        """)
        
        # Create indexes for common queries
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_cycle ON findings(cycle)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_findings_type ON findings(finding_type)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_from ON relationships(from_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_relationships_to ON relationships(to_id)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_cycle ON tasks(cycle)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_status ON research_questions(status)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_questions_area ON research_questions(area_id)")
        
        self.conn.commit()
    
    # =========================================================================
    # FINDINGS CRUD
    # =========================================================================
    
    def insert_finding(self, finding: Finding) -> str:
        """
        Insert a finding into the database.
        
        Args:
            finding: The Finding to insert
            
        Returns:
            The finding ID
        """
        self.conn.execute("""
            INSERT INTO findings (
                id, claim, finding_type, confidence, cycle,
                source_json, tags_json, evidence, statistical_support_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            finding.id,
            finding.claim,
            finding.finding_type if isinstance(finding.finding_type, str) else finding.finding_type.value,
            finding.confidence,
            finding.cycle,
            json.dumps(finding.source.model_dump()),
            json.dumps(finding.tags),
            finding.evidence,
            json.dumps(finding.statistical_support) if finding.statistical_support else None,
            finding.created_at.isoformat()
        ))
        self.conn.commit()
        return finding.id
    
    def get_finding(self, finding_id: str) -> Optional[Finding]:
        """Get a finding by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM findings WHERE id = ?", (finding_id,)
        )
        row = cursor.fetchone()
        return self._row_to_finding(row) if row else None
    
    def get_all_findings(self) -> List[Finding]:
        """Get all findings."""
        cursor = self.conn.execute("SELECT * FROM findings ORDER BY created_at")
        return [self._row_to_finding(row) for row in cursor.fetchall()]
    
    def get_findings_by_cycle(self, cycle: int) -> List[Finding]:
        """Get all findings from a specific cycle."""
        cursor = self.conn.execute(
            "SELECT * FROM findings WHERE cycle = ? ORDER BY created_at",
            (cycle,)
        )
        return [self._row_to_finding(row) for row in cursor.fetchall()]
    
    def get_findings_by_type(self, finding_type: str) -> List[Finding]:
        """Get all findings of a specific type."""
        cursor = self.conn.execute(
            "SELECT * FROM findings WHERE finding_type = ? ORDER BY created_at",
            (finding_type,)
        )
        return [self._row_to_finding(row) for row in cursor.fetchall()]
    
    def get_recent_findings(self, limit: int = 20) -> List[Finding]:
        """Get the most recent findings."""
        cursor = self.conn.execute(
            "SELECT * FROM findings ORDER BY created_at DESC LIMIT ?",
            (limit,)
        )
        return [self._row_to_finding(row) for row in cursor.fetchall()]
    
    def _row_to_finding(self, row: sqlite3.Row) -> Finding:
        """Convert a database row to a Finding model."""
        # Guard: finding_type may be NULL from a corrupted/old run
        ft = row["finding_type"] or "literature"
        try:
            finding_type = FindingType(ft)
        except ValueError:
            finding_type = FindingType.LITERATURE

        return Finding(
            id=row["id"],
            claim=row["claim"],
            finding_type=finding_type,
            confidence=row["confidence"],
            cycle=row["cycle"],
            source=Source(**json.loads(row["source_json"])),
            tags=json.loads(row["tags_json"]),
            evidence=row["evidence"],
            statistical_support=json.loads(row["statistical_support_json"]) if row["statistical_support_json"] else None,
            created_at=self._safe_datetime(row["created_at"])
        )
    
    # =========================================================================
    # RELATIONSHIPS CRUD
    # =========================================================================
    
    def insert_relationship(self, relationship: Relationship) -> int:
        """
        Insert a relationship into the database.
        
        Returns:
            The auto-generated relationship ID
        """
        cursor = self.conn.execute("""
            INSERT INTO relationships (
                from_id, to_id, relationship_type, strength, reasoning, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            relationship.from_id,
            relationship.to_id,
            relationship.relationship_type if isinstance(relationship.relationship_type, str) else relationship.relationship_type.value,
            relationship.strength,
            relationship.reasoning,
            relationship.created_at.isoformat()
        ))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_relationships_from(self, finding_id: str) -> List[Relationship]:
        """Get all relationships originating from a finding."""
        cursor = self.conn.execute(
            "SELECT * FROM relationships WHERE from_id = ?", (finding_id,)
        )
        return [self._row_to_relationship(row) for row in cursor.fetchall()]
    
    def get_relationships_to(self, finding_id: str) -> List[Relationship]:
        """Get all relationships pointing to a finding."""
        cursor = self.conn.execute(
            "SELECT * FROM relationships WHERE to_id = ?", (finding_id,)
        )
        return [self._row_to_relationship(row) for row in cursor.fetchall()]
    
    def get_all_relationships(self) -> List[Relationship]:
        """Get all relationships."""
        cursor = self.conn.execute("SELECT * FROM relationships")
        return [self._row_to_relationship(row) for row in cursor.fetchall()]
    
    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        """Convert a database row to a Relationship model."""
        return Relationship(
            id=row["id"],
            from_id=row["from_id"],
            to_id=row["to_id"],
            relationship_type=RelationshipType(row["relationship_type"]),
            strength=row["strength"],
            reasoning=row["reasoning"],
            created_at=self._safe_datetime(row["created_at"])
        )
    
    # =========================================================================
    # HYPOTHESES CRUD
    # =========================================================================
    
    def insert_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Insert a hypothesis into the database."""
        self.conn.execute("""
            INSERT INTO hypotheses (
                id, statement, status, cycle_proposed, cycle_resolved,
                supporting_finding_ids_json, contradicting_finding_ids_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            hypothesis.id,
            hypothesis.statement,
            hypothesis.status if isinstance(hypothesis.status, str) else hypothesis.status.value,
            hypothesis.cycle_proposed,
            hypothesis.cycle_resolved,
            json.dumps(hypothesis.supporting_finding_ids),
            json.dumps(hypothesis.contradicting_finding_ids),
            hypothesis.created_at.isoformat()
        ))
        self.conn.commit()
        return hypothesis.id
    
    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Get a hypothesis by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM hypotheses WHERE id = ?", (hypothesis_id,)
        )
        row = cursor.fetchone()
        return self._row_to_hypothesis(row) if row else None
    
    def get_all_hypotheses(self) -> List[Hypothesis]:
        """Get all hypotheses."""
        cursor = self.conn.execute("SELECT * FROM hypotheses ORDER BY created_at")
        return [self._row_to_hypothesis(row) for row in cursor.fetchall()]
    
    def get_hypotheses_by_status(self, status: str) -> List[Hypothesis]:
        """Get hypotheses by status."""
        cursor = self.conn.execute(
            "SELECT * FROM hypotheses WHERE status = ?", (status,)
        )
        return [self._row_to_hypothesis(row) for row in cursor.fetchall()]
    
    def update_hypothesis_status(
        self, 
        hypothesis_id: str, 
        status: str, 
        cycle_resolved: int = None
    ) -> None:
        """Update the status of a hypothesis."""
        if cycle_resolved:
            self.conn.execute(
                "UPDATE hypotheses SET status = ?, cycle_resolved = ? WHERE id = ?",
                (status, cycle_resolved, hypothesis_id)
            )
        else:
            self.conn.execute(
                "UPDATE hypotheses SET status = ? WHERE id = ?",
                (status, hypothesis_id)
            )
        self.conn.commit()
    
    def add_supporting_finding(self, hypothesis_id: str, finding_id: str) -> None:
        """Add a supporting finding to a hypothesis."""
        hypothesis = self.get_hypothesis(hypothesis_id)
        if hypothesis and finding_id not in hypothesis.supporting_finding_ids:
            hypothesis.supporting_finding_ids.append(finding_id)
            self.conn.execute(
                "UPDATE hypotheses SET supporting_finding_ids_json = ? WHERE id = ?",
                (json.dumps(hypothesis.supporting_finding_ids), hypothesis_id)
            )
            self.conn.commit()
    
    def _row_to_hypothesis(self, row: sqlite3.Row) -> Hypothesis:
        """Convert a database row to a Hypothesis model."""
        return Hypothesis(
            id=row["id"],
            statement=row["statement"],
            status=HypothesisStatus(row["status"]),
            cycle_proposed=row["cycle_proposed"],
            cycle_resolved=row["cycle_resolved"],
            supporting_finding_ids=json.loads(row["supporting_finding_ids_json"]),
            contradicting_finding_ids=json.loads(row["contradicting_finding_ids_json"]),
            created_at=datetime.fromisoformat(row["created_at"])
        )
    
    # =========================================================================
    # TASKS CRUD
    # =========================================================================
    
    def insert_task(self, task: Task) -> str:
        """Insert a task into the database."""
        self.conn.execute("""
            INSERT INTO tasks (
                id, task_type, description, goal, priority, status, cycle,
                result_finding_ids_json, error_message, execution_time_seconds,
                created_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task.id,
            task.task_type if isinstance(task.task_type, str) else task.task_type.value,
            task.description,
            task.goal,
            task.priority if isinstance(task.priority, str) else task.priority.value,
            task.status if isinstance(task.status, str) else task.status.value,
            task.cycle,
            json.dumps(task.result_finding_ids),
            task.error_message,
            task.execution_time_seconds,
            task.created_at.isoformat(),
            task.completed_at.isoformat() if task.completed_at else None
        ))
        self.conn.commit()
        return task.id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        cursor = self.conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cursor.fetchone()
        return self._row_to_task(row) if row else None
    
    def get_tasks_by_cycle(self, cycle: int) -> List[Task]:
        """Get all tasks from a specific cycle."""
        cursor = self.conn.execute(
            "SELECT * FROM tasks WHERE cycle = ? ORDER BY created_at",
            (cycle,)
        )
        return [self._row_to_task(row) for row in cursor.fetchall()]
    
    def update_task_status(
        self,
        task_id: str,
        status: str,
        error_message: str = None,
        execution_time: float = None,
        result_finding_ids: List[str] = None
    ) -> None:
        """Update task status and optionally other fields."""
        updates = ["status = ?"]
        params = [status]
        
        if status in ("completed", "failed"):
            updates.append("completed_at = ?")
            params.append(datetime.now().isoformat())
        
        if error_message is not None:
            updates.append("error_message = ?")
            params.append(error_message)
        
        if execution_time is not None:
            updates.append("execution_time_seconds = ?")
            params.append(execution_time)
        
        if result_finding_ids is not None:
            updates.append("result_finding_ids_json = ?")
            params.append(json.dumps(result_finding_ids))
        
        params.append(task_id)
        
        self.conn.execute(
            f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
            params
        )
        self.conn.commit()
    
    @staticmethod
    def _safe_datetime(value) -> datetime:
        """Safely parse a datetime string, returning now() if empty/None."""
        if not value:
            return datetime.now()
        try:
            # SQLite stores CURRENT_TIMESTAMP as '2026-02-20 01:54:40' (space not T)
            # Python 3.11+ handles this; for 3.10 we normalize the format
            return datetime.fromisoformat(str(value).replace(' ', 'T', 1))
        except (ValueError, TypeError):
            return datetime.now()

    def _row_to_task(self, row: sqlite3.Row) -> Task:
        """Convert a database row to a Task model."""
        return Task(
            id=row["id"],
            task_type=TaskType(row["task_type"]),
            description=row["description"],
            goal=row["goal"],
            priority=TaskPriority(row["priority"]),
            status=TaskStatus(row["status"]),
            cycle=row["cycle"],
            result_finding_ids=json.loads(row["result_finding_ids_json"]),
            error_message=row["error_message"],
            execution_time_seconds=row["execution_time_seconds"],
            created_at=self._safe_datetime(row["created_at"]),
            completed_at=self._safe_datetime(row["completed_at"]) if row["completed_at"] else None
        )
    
    # =========================================================================
    # RESEARCH QUESTIONS CRUD
    # =========================================================================
    
    def insert_question(self, question: ResearchQuestion) -> str:
        """Insert a research question into the database."""
        self.conn.execute("""
            INSERT INTO research_questions (
                id, question_text, status, priority, area_id, area_name,
                keywords_json, related_finding_ids_json, answer_summary,
                confidence_score, evidence_count, cycle_created, cycle_answered,
                created_at, answered_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            question.id,
            question.question_text,
            question.status if isinstance(question.status, str) else question.status.value,
            question.priority if isinstance(question.priority, str) else question.priority.value,
            question.area_id,
            question.area_name,
            json.dumps(question.keywords),
            json.dumps(question.related_finding_ids),
            question.answer_summary,
            question.confidence_score,
            question.evidence_count,
            question.cycle_created,
            question.cycle_answered,
            question.created_at.isoformat(),
            question.answered_at.isoformat() if question.answered_at else None
        ))
        self.conn.commit()
        return question.id
    
    def get_question(self, question_id: str) -> Optional[ResearchQuestion]:
        """Get a research question by ID."""
        cursor = self.conn.execute(
            "SELECT * FROM research_questions WHERE id = ?", (question_id,)
        )
        row = cursor.fetchone()
        return self._row_to_question(row) if row else None
    
    def get_all_questions(self) -> List[ResearchQuestion]:
        """Get all research questions."""
        cursor = self.conn.execute(
            "SELECT * FROM research_questions ORDER BY created_at"
        )
        return [self._row_to_question(row) for row in cursor.fetchall()]
    
    def get_questions_by_status(self, status: str) -> List[ResearchQuestion]:
        """Get research questions by status."""
        cursor = self.conn.execute(
            "SELECT * FROM research_questions WHERE status = ? ORDER BY priority DESC, created_at",
            (status,)
        )
        return [self._row_to_question(row) for row in cursor.fetchall()]
    
    def get_unanswered_questions(self) -> List[ResearchQuestion]:
        """Get all unanswered or partially answered questions."""
        cursor = self.conn.execute(
            "SELECT * FROM research_questions WHERE status IN ('unanswered', 'partial') ORDER BY priority DESC, created_at"
        )
        return [self._row_to_question(row) for row in cursor.fetchall()]
    
    def get_questions_by_area(self, area_id: str) -> List[ResearchQuestion]:
        """Get all questions for a specific research area."""
        cursor = self.conn.execute(
            "SELECT * FROM research_questions WHERE area_id = ? ORDER BY created_at",
            (area_id,)
        )
        return [self._row_to_question(row) for row in cursor.fetchall()]
    
    def update_question_status(
        self,
        question_id: str,
        status: str = None,
        answer_summary: str = None,
        confidence_score: float = None,
        evidence_count: int = None,
        related_finding_ids: List[str] = None,
        cycle_answered: int = None
    ) -> None:
        """Update question status and answer information."""
        updates = []
        params = []
        
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            
            if status == "answered":
                updates.append("answered_at = ?")
                params.append(datetime.now().isoformat())
                if cycle_answered:
                    updates.append("cycle_answered = ?")
                    params.append(cycle_answered)
        
        if answer_summary is not None:
            updates.append("answer_summary = ?")
            params.append(answer_summary)
        
        if confidence_score is not None:
            updates.append("confidence_score = ?")
            params.append(confidence_score)
        
        if evidence_count is not None:
            updates.append("evidence_count = ?")
            params.append(evidence_count)
        
        if related_finding_ids is not None:
            updates.append("related_finding_ids_json = ?")
            params.append(json.dumps(related_finding_ids))
        
        if not updates:
            return
        
        params.append(question_id)
        
        self.conn.execute(
            f"UPDATE research_questions SET {', '.join(updates)} WHERE id = ?",
            params
        )
        self.conn.commit()
    
    def add_finding_to_question(self, question_id: str, finding_id: str) -> None:
        """Link a finding to a question."""
        question = self.get_question(question_id)
        if question and finding_id not in question.related_finding_ids:
            question.related_finding_ids.append(finding_id)
            self.conn.execute(
                "UPDATE research_questions SET related_finding_ids_json = ?, evidence_count = ? WHERE id = ?",
                (json.dumps(question.related_finding_ids), len(question.related_finding_ids), question_id)
            )
            self.conn.commit()
    
    def _row_to_question(self, row: sqlite3.Row) -> ResearchQuestion:
        """Convert a database row to a ResearchQuestion model."""
        return ResearchQuestion(
            id=row["id"],
            question_text=row["question_text"],
            status=QuestionStatus(row["status"]),
            priority=TaskPriority(row["priority"]),
            area_id=row["area_id"],
            area_name=row["area_name"],
            keywords=json.loads(row["keywords_json"]) if row["keywords_json"] else [],
            related_finding_ids=json.loads(row["related_finding_ids_json"]) if row["related_finding_ids_json"] else [],
            answer_summary=row["answer_summary"],
            confidence_score=row["confidence_score"] or 0.0,
            evidence_count=row["evidence_count"] or 0,
            cycle_created=row["cycle_created"] or 1,
            cycle_answered=row["cycle_answered"],
            created_at=self._safe_datetime(row["created_at"]),
            answered_at=self._safe_datetime(row["answered_at"]) if row["answered_at"] else None
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_statistics(self) -> dict:
        """Get summary statistics about the database."""
        stats = {}
        
        # Count findings
        cursor = self.conn.execute("SELECT COUNT(*) FROM findings")
        stats["total_findings"] = cursor.fetchone()[0]
        
        # Count by type
        cursor = self.conn.execute(
            "SELECT finding_type, COUNT(*) FROM findings GROUP BY finding_type"
        )
        stats["findings_by_type"] = dict(cursor.fetchall())
        
        # Count relationships
        cursor = self.conn.execute("SELECT COUNT(*) FROM relationships")
        stats["total_relationships"] = cursor.fetchone()[0]
        
        # Count hypotheses by status
        cursor = self.conn.execute(
            "SELECT status, COUNT(*) FROM hypotheses GROUP BY status"
        )
        stats["hypotheses_by_status"] = dict(cursor.fetchall())
        
        # Count tasks by status
        cursor = self.conn.execute(
            "SELECT status, COUNT(*) FROM tasks GROUP BY status"
        )
        stats["tasks_by_status"] = dict(cursor.fetchall())
        
        # Get max cycle
        cursor = self.conn.execute("SELECT MAX(cycle) FROM findings")
        result = cursor.fetchone()[0]
        stats["current_cycle"] = result if result else 0
        
        # Count research questions by status
        try:
            cursor = self.conn.execute(
                "SELECT status, COUNT(*) FROM research_questions GROUP BY status"
            )
            stats["questions_by_status"] = dict(cursor.fetchall())
            
            cursor = self.conn.execute("SELECT COUNT(*) FROM research_questions")
            stats["total_questions"] = cursor.fetchone()[0]
        except Exception:
            # Table might not exist in older databases
            stats["questions_by_status"] = {}
            stats["total_questions"] = 0
        
        return stats
    
    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
