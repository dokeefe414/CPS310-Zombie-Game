import psycopg2
from psycopg2.extras import execute_values


class DatabaseManager:
    """Manages connection, schema creation, and bulk batch logging for simulation runs."""

    def __init__(
        self,
        dbname: str = "zombie_db",
        user: str = "zombie_admin",
        password: str = "zombiepass123",
        host: str = "127.0.0.1",
        port: int = 5432,
    ):
        self.config = {
            "dbname": dbname,
            "user": user,
            "password": password,
            "host": host,
            "port": port,
        }
        self.init_schema()

    def get_connection(self):
        return psycopg2.connect(**self.config)

    def init_schema(self):
        create_runs_table = """
        CREATE TABLE IF NOT EXISTS simulation_runs (
            run_id SERIAL PRIMARY KEY,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            initial_regular_humans INT,
            initial_armed_humans INT,
            initial_zombies INT,
            total_ticks INT,
            outcome VARCHAR(50)
        );
        """
        create_snapshots_table = """
        CREATE TABLE IF NOT EXISTS population_snapshots (
            snapshot_id SERIAL PRIMARY KEY,
            run_id INT REFERENCES simulation_runs(run_id) ON DELETE CASCADE,
            tick_number INT,
            human_count INT,
            zombie_count INT
        );
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_runs_table)
                    cur.execute(create_snapshots_table)
                conn.commit()
            print("[DatabaseManager] Database initialized and tables verified.")
        except Exception as e:
            print(f"[DatabaseManager] Setup failed: {e}")

    def save_run(self, metadata: dict, history_records: list[tuple]):
        insert_run_sql = """
        INSERT INTO simulation_runs (
            initial_regular_humans, initial_armed_humans, initial_zombies, total_ticks, outcome
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING run_id;
        """
        insert_snapshots_sql = """
        INSERT INTO population_snapshots (run_id, tick_number, human_count, zombie_count)
        VALUES %s;
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_run_sql,
                        (
                            metadata["initial_regular"],
                            metadata["initial_armed"],
                            metadata["initial_zombies"],
                            metadata["total_ticks"],
                            metadata["outcome"],
                        ),
                    )
                    run_id = cur.fetchone()[0]

                    # Downsample historical ticks to prevent blocking I/O while preserving curve trends
                    stride = max(1, len(history_records) // 300)
                    batch_data = [
                        (run_id, tick, h_cnt, z_cnt)
                        for tick, h_cnt, z_cnt in history_records[::stride]
                    ]

                    execute_values(cur, insert_snapshots_sql, batch_data)
                conn.commit()
            print(f"[DatabaseManager] Successfully logged Run #{run_id} ({len(batch_data)} rows) to PostgreSQL.")
        except Exception as e:
            print(f"[DatabaseManager] Failed to write run to database: {e}")