"""
Oracle AI Vector Search persistence layer for discover-github.

Stores repository metadata with vector embeddings, enabling:
- Semantic deduplication (don't re-cover similar repos)
- Trend trajectory tracking over time
- "Find similar repos" queries
- Historical analytics and category analysis
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import oracledb

logger = logging.getLogger(__name__)

# Schema version for migrations
SCHEMA_VERSION = 1

DDL_STATEMENTS = [
    # Core repository table with vector embedding
    """
    CREATE TABLE repositories (
        id               NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        full_name        VARCHAR2(500) NOT NULL UNIQUE,
        owner            VARCHAR2(255) NOT NULL,
        repo_name        VARCHAR2(255) NOT NULL,
        description      VARCHAR2(4000),
        url              VARCHAR2(1000),
        homepage         VARCHAR2(1000),
        language         VARCHAR2(100),
        topics           CLOB,
        stars            NUMBER DEFAULT 0,
        forks            NUMBER DEFAULT 0,
        open_issues      NUMBER DEFAULT 0,
        watchers         NUMBER DEFAULT 0,
        subscribers      NUMBER DEFAULT 0,
        contributor_count NUMBER DEFAULT 0,
        recent_commit_count NUMBER DEFAULT 0,
        has_ci           NUMBER(1) DEFAULT 0,
        license_name     VARCHAR2(255),
        is_fork          NUMBER(1) DEFAULT 0,
        is_archived      NUMBER(1) DEFAULT 0,
        size_kb          NUMBER DEFAULT 0,
        default_branch   VARCHAR2(100) DEFAULT 'main',
        repo_created_at  TIMESTAMP WITH TIME ZONE,
        repo_pushed_at   TIMESTAMP WITH TIME ZONE,
        repo_updated_at  TIMESTAMP WITH TIME ZONE,
        first_seen_at    TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
        last_updated_at  TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
        readme_content   CLOB,
        embedding        VECTOR(384, FLOAT32)
    )
    """,

    # Snapshot table for tracking metrics over time
    """
    CREATE TABLE repo_snapshots (
        id               NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        repo_id          NUMBER NOT NULL REFERENCES repositories(id),
        snapshot_date    DATE NOT NULL,
        stars            NUMBER DEFAULT 0,
        forks            NUMBER DEFAULT 0,
        open_issues      NUMBER DEFAULT 0,
        contributor_count NUMBER DEFAULT 0,
        recent_commit_count NUMBER DEFAULT 0,
        breakout_score   NUMBER(10,4) DEFAULT 0,
        CONSTRAINT uq_repo_snapshot UNIQUE (repo_id, snapshot_date)
    )
    """,

    # Generated content table
    """
    CREATE TABLE generated_content (
        id               NUMBER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
        repo_id          NUMBER NOT NULL REFERENCES repositories(id),
        content_type     VARCHAR2(50) NOT NULL,
        content_text     CLOB NOT NULL,
        platform         VARCHAR2(50),
        generated_at     TIMESTAMP WITH TIME ZONE DEFAULT SYSTIMESTAMP,
        published        NUMBER(1) DEFAULT 0,
        published_at     TIMESTAMP WITH TIME ZONE
    )
    """,

    # Category classifications
    """
    CREATE TABLE repo_categories (
        repo_id          NUMBER NOT NULL REFERENCES repositories(id),
        category         VARCHAR2(100) NOT NULL,
        confidence       NUMBER(5,4) DEFAULT 1.0,
        CONSTRAINT pk_repo_cat PRIMARY KEY (repo_id, category)
    )
    """,

    # Vector similarity index for fast nearest-neighbor search
    """
    CREATE VECTOR INDEX idx_repo_embedding ON repositories(embedding)
    ORGANIZATION NEIGHBOR PARTITIONS
    DISTANCE COSINE
    WITH TARGET ACCURACY 95
    """,

    # Performance indexes
    "CREATE INDEX idx_repo_stars ON repositories(stars DESC)",
    "CREATE INDEX idx_repo_language ON repositories(language)",
    "CREATE INDEX idx_repo_first_seen ON repositories(first_seen_at DESC)",
    "CREATE INDEX idx_snapshot_date ON repo_snapshots(snapshot_date DESC)",
    "CREATE INDEX idx_content_type ON generated_content(content_type, platform)",
]


class DiscoverDB:
    """Oracle AI Vector Search persistence for discovered repositories."""

    def __init__(self, user: str, password: str, dsn: str):
        self.user = user
        self.password = password
        self.dsn = dsn
        self._pool = None

    def connect(self) -> oracledb.Connection:
        """Get a connection (creates pool on first call)."""
        if self._pool is None:
            self._pool = oracledb.create_pool(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=1,
                max=5,
                increment=1,
            )
        return self._pool.acquire()

    def initialize_schema(self):
        """Create tables and indexes if they don't exist."""
        conn = self.connect()
        cur = conn.cursor()

        for ddl in DDL_STATEMENTS:
            try:
                cur.execute(ddl)
                logger.debug(f"Executed: {ddl[:60]}...")
            except oracledb.DatabaseError as e:
                error_code = e.args[0].code if e.args else 0
                # ORA-00955: name already exists — skip
                # ORA-01408: index column list already indexed
                if error_code in (955, 1408):
                    continue
                logger.warning(f"DDL warning ({error_code}): {e}")

        conn.commit()
        conn.close()
        logger.info("Schema initialized")

    def upsert_repo(self, meta, embedding: Optional[list[float]] = None) -> int:
        """
        Insert or update a repository. Returns the repo ID.
        If the repo already exists, updates metrics and last_updated_at.
        """
        conn = self.connect()
        cur = conn.cursor()

        topics_json = json.dumps(meta.topics) if meta.topics else "[]"

        # Check if repo exists
        cur.execute(
            "SELECT id FROM repositories WHERE full_name = :1",
            [meta.full_name],
        )
        row = cur.fetchone()

        if row:
            repo_id = row[0]
            # Update scalar columns first
            cur.execute("""
                UPDATE repositories SET
                    description = :1, stars = :2, forks = :3,
                    open_issues = :4, watchers = :5, subscribers = :6,
                    contributor_count = :7, recent_commit_count = :8,
                    has_ci = :9, language = :10,
                    license_name = :11, is_archived = :12,
                    repo_pushed_at = :13, repo_updated_at = :14,
                    last_updated_at = SYSTIMESTAMP,
                    size_kb = :15
                WHERE id = :16
            """, [
                meta.description, meta.stars, meta.forks,
                meta.open_issues, meta.watchers, meta.subscribers,
                meta.contributor_count, meta.recent_commit_count,
                1 if meta.has_ci else 0, meta.language,
                meta.license_name, 1 if meta.is_archived else 0,
                meta.pushed_at, meta.updated_at,
                meta.size_kb,
                repo_id,
            ])

            # Update LOB/VECTOR columns separately (ORA-24816 workaround)
            cur.execute(
                "UPDATE repositories SET topics = :1 WHERE id = :2",
                [topics_json, repo_id],
            )
            if meta.readme_content:
                cur.execute(
                    "UPDATE repositories SET readme_content = :1 WHERE id = :2",
                    [meta.readme_content, repo_id],
                )
            if embedding:
                cur.execute(
                    "UPDATE repositories SET embedding = :1 WHERE id = :2",
                    [str(embedding), repo_id],
                )
        else:
            # Insert new repo (scalar columns only, then update LOB/VECTOR)
            repo_id_var = cur.var(int)
            cur.execute("""
                INSERT INTO repositories (
                    full_name, owner, repo_name, description, url, homepage,
                    language, stars, forks, open_issues, watchers,
                    subscribers, contributor_count, recent_commit_count,
                    has_ci, license_name, is_fork, is_archived, size_kb,
                    default_branch, repo_created_at, repo_pushed_at,
                    repo_updated_at
                ) VALUES (
                    :1, :2, :3, :4, :5, :6,
                    :7, :8, :9, :10, :11,
                    :12, :13, :14,
                    :15, :16, :17, :18, :19,
                    :20, :21, :22,
                    :23
                ) RETURNING id INTO :24
            """, [
                meta.full_name, meta.owner, meta.name,
                meta.description, meta.url, meta.homepage,
                meta.language,
                meta.stars, meta.forks, meta.open_issues, meta.watchers,
                meta.subscribers, meta.contributor_count, meta.recent_commit_count,
                1 if meta.has_ci else 0, meta.license_name,
                1 if meta.is_fork else 0, 1 if meta.is_archived else 0,
                meta.size_kb, meta.default_branch,
                meta.created_at, meta.pushed_at, meta.updated_at,
                repo_id_var,
            ])
            repo_id = repo_id_var.getvalue()[0]

            # Update LOB/VECTOR columns separately (ORA-24816 workaround)
            cur.execute(
                "UPDATE repositories SET topics = :1 WHERE id = :2",
                [topics_json, repo_id],
            )
            if meta.readme_content:
                cur.execute(
                    "UPDATE repositories SET readme_content = :1 WHERE id = :2",
                    [meta.readme_content, repo_id],
                )
            if embedding:
                cur.execute(
                    "UPDATE repositories SET embedding = :1 WHERE id = :2",
                    [str(embedding), repo_id],
                )

        conn.commit()
        conn.close()
        return repo_id

    def record_snapshot(self, repo_id: int, breakout_score: float = 0.0):
        """Record today's metrics snapshot for trend tracking."""
        conn = self.connect()
        cur = conn.cursor()

        try:
            cur.execute("""
                INSERT INTO repo_snapshots (
                    repo_id, snapshot_date, stars, forks, open_issues,
                    contributor_count, recent_commit_count, breakout_score
                )
                SELECT :1, TRUNC(SYSDATE), stars, forks, open_issues,
                       contributor_count, recent_commit_count, :2
                FROM repositories WHERE id = :3
            """, [repo_id, breakout_score, repo_id])
            conn.commit()
        except oracledb.DatabaseError as e:
            # Unique constraint violation = already snapshotted today
            if e.args[0].code == 1:
                logger.debug(f"Snapshot already exists for repo {repo_id} today")
            else:
                raise
        finally:
            conn.close()

    def find_similar(
        self,
        embedding: list[float],
        limit: int = 5,
        min_similarity: float = 0.7,
        exclude_full_name: Optional[str] = None,
    ) -> list[dict]:
        """
        Find semantically similar repos using vector cosine similarity.
        Returns list of {full_name, similarity, description, stars}.
        """
        conn = self.connect()
        cur = conn.cursor()

        exclude_clause = ""
        params = {"emb": str(embedding), "lim": limit}
        if exclude_full_name:
            exclude_clause = "WHERE full_name != :excl"
            params["excl"] = exclude_full_name

        cur.execute(f"""
            SELECT full_name, description, stars, language,
                   VECTOR_DISTANCE(embedding, :emb, COSINE) AS distance
            FROM repositories
            {exclude_clause}
            ORDER BY VECTOR_DISTANCE(embedding, :emb, COSINE)
            FETCH FIRST :lim ROWS ONLY
        """, params)

        results = []
        for row in cur.fetchall():
            similarity = 1 - row[4]  # cosine distance to similarity
            if similarity >= min_similarity:
                results.append({
                    "full_name": row[0],
                    "description": row[1],
                    "stars": row[2],
                    "language": row[3],
                    "similarity": round(similarity, 4),
                })

        conn.close()
        return results

    def is_duplicate(
        self,
        embedding: list[float],
        threshold: float = 0.92,
        exclude_full_name: Optional[str] = None,
    ) -> Optional[str]:
        """
        Check if a repo is semantically too similar to one already stored.
        Returns the full_name of the duplicate if found, else None.
        """
        similar = self.find_similar(
            embedding, limit=1, min_similarity=threshold,
            exclude_full_name=exclude_full_name,
        )
        if similar:
            return similar[0]["full_name"]
        return None

    def save_content(
        self,
        repo_id: int,
        content_type: str,
        content_text: str,
        platform: Optional[str] = None,
    ) -> int:
        """Save generated content (summary, analysis, social post)."""
        conn = self.connect()
        cur = conn.cursor()

        content_id_var = cur.var(int)
        cur.execute("""
            INSERT INTO generated_content (
                repo_id, content_type, content_text, platform
            ) VALUES (:1, :2, :3, :4)
            RETURNING id INTO :5
        """, [repo_id, content_type, content_text, platform, content_id_var])

        conn.commit()
        content_id = content_id_var.getvalue()[0]
        conn.close()
        return content_id

    def save_categories(self, repo_id: int, categories: list[dict]):
        """Save category classifications for a repo."""
        conn = self.connect()
        cur = conn.cursor()

        for cat in categories:
            try:
                cur.execute("""
                    MERGE INTO repo_categories rc
                    USING DUAL ON (rc.repo_id = :1 AND rc.category = :2)
                    WHEN MATCHED THEN UPDATE SET confidence = :3
                    WHEN NOT MATCHED THEN INSERT (repo_id, category, confidence)
                        VALUES (:1, :2, :3)
                """, [repo_id, cat["category"], cat.get("confidence", 1.0)])
            except oracledb.DatabaseError as e:
                logger.warning(f"Error saving category: {e}")

        conn.commit()
        conn.close()

    def get_star_velocity(self, full_name: str, days: int = 7) -> Optional[float]:
        """
        Calculate star velocity (stars gained per day) by comparing
        current stars to the oldest snapshot within the window.
        """
        conn = self.connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT r.stars - NVL(s.stars, r.stars),
                   GREATEST(TRUNC(SYSDATE) - NVL(s.snapshot_date, TRUNC(SYSDATE)), 1)
            FROM repositories r
            LEFT JOIN (
                SELECT repo_id, stars, snapshot_date
                FROM repo_snapshots
                WHERE snapshot_date >= TRUNC(SYSDATE) - :1
                ORDER BY snapshot_date ASC
                FETCH FIRST 1 ROW ONLY
            ) s ON s.repo_id = r.id
            WHERE r.full_name = :2
        """, [days, full_name])

        row = cur.fetchone()
        conn.close()

        if row and row[1] > 0:
            return row[0] / row[1]
        return None

    def get_trending_categories(self, days: int = 7, limit: int = 10) -> list[dict]:
        """Get most common categories among recently discovered repos."""
        conn = self.connect()
        cur = conn.cursor()

        cur.execute("""
            SELECT rc.category, COUNT(*) AS cnt,
                   ROUND(AVG(r.stars)) AS avg_stars
            FROM repo_categories rc
            JOIN repositories r ON r.id = rc.repo_id
            WHERE r.first_seen_at >= SYSTIMESTAMP - INTERVAL :1 DAY
            GROUP BY rc.category
            ORDER BY cnt DESC
            FETCH FIRST :2 ROWS ONLY
        """, [days, limit])

        results = []
        for row in cur.fetchall():
            results.append({
                "category": row[0],
                "count": row[1],
                "avg_stars": row[2],
            })

        conn.close()
        return results

    def get_repo_count(self) -> int:
        """Total repos in the database."""
        conn = self.connect()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM repositories")
        count = cur.fetchone()[0]
        conn.close()
        return count

    def close(self):
        if self._pool:
            self._pool.close()
