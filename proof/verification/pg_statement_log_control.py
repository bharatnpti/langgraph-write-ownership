"""Turn Postgres statement logging on/off for one database, and mark log offsets.

Provenance: copied VERBATIM (not one character changed) from the adversarial-
review scratchpad as `logctl.py`. No path fix was needed: this script already
hardcodes this repo's absolute path.

CAUTION -- this is a control utility, not a probe: `main()` runs
`ALTER DATABASE ... SET log_statement = 'all'`, a PERSISTENT, cluster-level
config change on whatever database you name, against the SHARED embedded
Postgres instance at spike/pgdata. It stays in effect until you run this
script again with "off" against the same database. It was re-run here only
as `on <throwaway db>` immediately followed by `off <same db>`, to confirm
the code path still works, and every database it touched that way was left
back in its default (unlogged) state -- see ../README.md. Point it at a real
scenario's database only if you deliberately want that database's statements
logged.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "spike"))
import psycopg
from _harness import pg_conninfo

LOG = pathlib.Path(
    "spike/pgdata/log")
MARK = pathlib.Path(__file__).resolve().parent / "logoffset.txt"


def main() -> None:
    what, db = sys.argv[1], sys.argv[2]
    pg_conninfo(db)  # ensure the database exists
    admin = pg_conninfo("postgres").replace("/postgres?", "/postgres?")
    with psycopg.connect(admin, autocommit=True) as c:
        if what == "on":
            c.execute(f'alter database "{db}" set log_statement = \'all\'')
            c.execute(f'alter database "{db}" set log_parameter_max_length = 140')
            MARK.write_text(str(LOG.stat().st_size))
            print("logging ON for", db, "log offset", MARK.read_text())
        else:
            for p in ("log_statement", "log_parameter_max_length", "log_min_messages"):
                c.execute(f'alter database "{db}" reset {p}')
            print("logging OFF for", db)
        row = c.execute(
            "select setconfig from pg_db_role_setting s join pg_database d"
            " on d.oid = s.setdatabase where d.datname = %s", (db,)).fetchone()
        print("pg_db_role_setting now:", row)


if __name__ == "__main__":
    main()
