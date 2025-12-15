-- Reset script: drop all experimental objects and recreate schema/views.
-- Usage:
--   sqlite3 <db_path> < experiments/sql/reset.sql
--
-- This file is intentionally self-contained (no .read) to make execution predictable.

PRAGMA foreign_keys = OFF;

DROP VIEW IF EXISTS v_items_latest;
DROP VIEW IF EXISTS v_item_status_latest;

DROP TABLE IF EXISTS item_status_history;
DROP TABLE IF EXISTS items;
DROP TABLE IF EXISTS entity_aliases;
DROP TABLE IF EXISTS projects;
DROP TABLE IF EXISTS people;

PRAGMA foreign_keys = ON;

-- Recreate schema

CREATE TABLE IF NOT EXISTS people (
  person_id INTEGER PRIMARY KEY AUTOINCREMENT,
  handle TEXT NOT NULL UNIQUE,
  display_name TEXT,
  team TEXT,
  role TEXT,
  created_ts INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS projects (
  project_id INTEGER PRIMARY KEY AUTOINCREMENT,
  code TEXT UNIQUE,
  name TEXT NOT NULL,
  owner_person_id INTEGER,
  status TEXT NOT NULL DEFAULT 'ACTIVE',
  created_ts INTEGER NOT NULL,
  FOREIGN KEY(owner_person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS entity_aliases (
  alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
  entity_type TEXT NOT NULL,
  entity_ref TEXT NOT NULL,
  alias TEXT NOT NULL,
  source TEXT DEFAULT 'seed',
  UNIQUE(entity_type, alias)
);

CREATE TABLE IF NOT EXISTS items (
  item_id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  item_type TEXT NOT NULL DEFAULT 'TASK',
  owner_person_id INTEGER,
  project_id INTEGER,
  priority INTEGER NOT NULL DEFAULT 3,
  tags TEXT,
  created_ts INTEGER NOT NULL,
  due_ts INTEGER,
  description TEXT,
  is_deleted INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(owner_person_id) REFERENCES people(person_id),
  FOREIGN KEY(project_id) REFERENCES projects(project_id)
);

CREATE TABLE IF NOT EXISTS item_status_history (
  status_id INTEGER PRIMARY KEY AUTOINCREMENT,
  item_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  note TEXT,
  changed_ts INTEGER NOT NULL,
  changed_by_person_id INTEGER,
  FOREIGN KEY(item_id) REFERENCES items(item_id),
  FOREIGN KEY(changed_by_person_id) REFERENCES people(person_id)
);

CREATE INDEX IF NOT EXISTS idx_items_owner ON items(owner_person_id);
CREATE INDEX IF NOT EXISTS idx_items_project ON items(project_id);
CREATE INDEX IF NOT EXISTS idx_items_due_ts ON items(due_ts);
CREATE INDEX IF NOT EXISTS idx_items_created_ts ON items(created_ts);
CREATE INDEX IF NOT EXISTS idx_status_item_ts ON item_status_history(item_id, changed_ts DESC);
CREATE INDEX IF NOT EXISTS idx_status_status ON item_status_history(status);
CREATE INDEX IF NOT EXISTS idx_alias_entity ON entity_aliases(entity_type, entity_ref);
CREATE UNIQUE INDEX IF NOT EXISTS idx_alias_unique ON entity_aliases(entity_type, alias);

CREATE VIEW IF NOT EXISTS v_item_status_latest AS
SELECT s.*
FROM item_status_history s
JOIN (
  SELECT item_id, MAX(changed_ts) AS max_changed_ts
  FROM item_status_history
  GROUP BY item_id
) latest
ON latest.item_id = s.item_id AND latest.max_changed_ts = s.changed_ts;

CREATE VIEW IF NOT EXISTS v_items_latest AS
SELECT
  i.item_id,
  i.title,
  i.item_type,
  p.person_id AS owner_id,
  p.handle AS owner,
  p.display_name AS owner_display_name,
  p.team AS owner_team,
  p.role AS owner_role,
  pr.project_id,
  pr.code AS project_code,
  pr.name AS project_name,
  pr.status AS project_status,
  st.status AS status,
  st.note AS status_note,
  st.changed_ts AS status_ts,
  i.priority,
  i.tags,
  i.created_ts,
  i.due_ts,
  i.description
FROM items i
LEFT JOIN people p ON p.person_id = i.owner_person_id
LEFT JOIN projects pr ON pr.project_id = i.project_id
LEFT JOIN v_item_status_latest st ON st.item_id = i.item_id
WHERE i.is_deleted = 0;
