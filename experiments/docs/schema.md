# Experimental SQLite Schema (Multi-table + Views)

This schema is designed for *reproducible* NL->IR->SQL experiments on task/issue/"item" style question answering.
It separates common business dimensions (people, projects) from the core fact (items) and models status transitions
as append-only history. For most queries, the recommended entry point is the external view `v_items_latest`, which
behaves like a single denormalized table.

## Design goals

- **Generic naming**: uses `items`, `people`, `projects`, `item_status_history` to stay domain-agnostic.
- **Explainability**: fields map cleanly to natural-language concepts (owner, project, status, due, priority, tags).
- **Join-friendly**: the most common joins are encapsulated into views so that baselines can query one view.
- **Low complexity**: no cascading foreign keys and no deep normalization requirements.

## Tables

### `people`

Represents humans (or assignees) who can own items and/or perform status changes.

Fields:
- `person_id`: surrogate primary key.
- `handle`: canonical identifier for retrieval and matching (unique).
- `display_name`: human readable name.
- `team`: department/team label (stored as TEXT for simplicity).
- `role`: job role/title.
- `created_ts`: creation time in epoch milliseconds.

### `projects`

Represents a project dimension that items may belong to.

Fields:
- `project_id`: surrogate primary key.
- `code`: short code (unique, optional), suitable for NL mentions.
- `name`: project name.
- `owner_person_id`: optional owner (FK to `people.person_id`).
- `status`: project status label (`ACTIVE` by default).
- `created_ts`: creation time in epoch milliseconds.

### `entity_aliases`

Lightweight alias dictionary used by KG-lite / IR. It ties canonical entities to a handful of alternate surface
forms so prompts and retrieval have deterministic coverage.

Fields:
- `alias_id`: surrogate primary key.
- `entity_type`: `PERSON`, `PROJECT`, or `TAG`.
- `entity_ref`: canonical reference (person handle, project code, tag name).
- `alias`: alternate surface form (unique per entity type).
- `source`: provenance label (default `seed`).

### `items`

The core unit for question answering: task, issue, request, or any actionable item.

Fields:
- `item_id`: surrogate primary key.
- `title`: short title (the "task/issue name").
- `item_type`: coarse type label (`TASK` by default).
- `owner_person_id`: current owner/assignee (FK to `people.person_id`).
- `project_id`: optional project association (FK to `projects.project_id`).
- `priority`: integer priority (1=highest).
- `tags`: comma-separated tags (kept denormalized to limit table count and keep SQL simple).
- `created_ts`: creation time in epoch milliseconds.
- `due_ts`: due time in epoch milliseconds (nullable).
- `description`: optional longer description.
- `is_deleted`: soft-delete flag (0/1); views filter out deleted items.

### `item_status_history`

Append-only history of status changes. "Current status" is derived as the latest row by `changed_ts`.

Fields:
- `status_id`: surrogate primary key.
- `item_id`: item reference (FK to `items.item_id`).
- `status`: status label (e.g., `TODO`, `IN_PROGRESS`, `BLOCKED`, `DONE`, `CANCELED`).
- `note`: optional note explaining the status.
- `changed_ts`: when the change happened (epoch milliseconds).
- `changed_by_person_id`: optional actor (FK to `people.person_id`).

## Views

### `v_item_status_latest`

Returns the latest status-history row *per item* using `MAX(changed_ts)` grouped by `item_id`.
This is a helper view used by `v_items_latest`.

Columns:
- Same as `item_status_history` (`status_id`, `item_id`, `status`, `note`, `changed_ts`, `changed_by_person_id`).

### `v_items_latest` (external experimental view)

This is the "single-table output" view used for experiments. It joins the most common dimensions and selects
the latest status, producing a denormalized row per item.

Key columns (recommended for NL2SQL baselines):
- **Owner**: `owner`, `owner_display_name`, `owner_team`
- **Project**: `project_code`, `project_name`, `project_status`
- **Status**: `status`, `status_note`, `status_ts`
- **Item properties**: `title`, `item_type`, `priority`, `tags`, `created_ts`, `due_ts`

Notes:
- Uses `LEFT JOIN` to keep items visible even if some dimensions are missing.
- Filters out soft-deleted items via `WHERE is_deleted = 0`.

## How to reset/recreate

- Reset everything: `sqlite3 <db_path> < experiments/sql/reset.sql`
- Inspect the external view schema: `PRAGMA table_info(v_items_latest);`
- Deterministic seeding: `python experiments/runner.py seed --seed 42 --items 500` (writes DB + `experiments/artifacts/seed_summary.json` and populates alias rows).
