# Imara – Project Setup & Development Guide

Dieses Dokument beschreibt das grundlegende Setup sowie die wichtigsten Entwicklungs- und Arbeitsabläufe für das **Imara-Projekt**.

---

## Required Tools

Bevor du startest, installiere folgende Werkzeuge:

* **uv** – schnelles Python-Package- und Environment-Management
  [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
* **dvc** – Data Version Control für reproduzierbare ML-Pipelines
  [https://dvc.org/doc/install](https://dvc.org/doc/install)
* **git** – Versionskontrolle
* **Python 3.10+**

Optional, aber empfohlen:

* **pre-commit**
* **pytest**, **ruff**, **mypy** (werden durch `uv sync` installiert)

---

## Setup Instructions

### Repository klonen

```bash
git clone https://github.com/FH-OST-Metis/imara.git
cd imara
```

### Abhängigkeiten installieren

```bash
uv sync
```

Dies erstellt ein virtuelles Environment und installiert alle im `pyproject.toml` definierten Pakete.

---

### Pre-commit Hooks installieren

```bash
uv run pre-commit install
```

Pre-commit wird danach automatisch bei jedem Commit ausgeführt.

---

### S3-Authentifizierung

1. Erstelle einen neuen Access Key im
   [https://supabase.com/dashboard/project/hjijyloqvddflojzrvcn/storage/s3](https://supabase.com/dashboard/project/hjijyloqvddflojzrvcn/storage/s3)

2. Kopiere das Template und erstelle deine lokale Config:

   ```bash
   cp .dvc/config.local-template .dvc/config.local
   ```

3. Fülle `access_key_id` und `secret_access_key` aus Schritt 1 in `.dvc/config.local` ein.

Hinweis: Die Datei `.dvc/config.local` wird **nicht** versioniert.

---

## CI/CD Tools

### Linter (ruff)

```bash
uv run ruff check .
```

### Formatter

```bash
uv run ruff format .
```

### Type Checking (mypy)

```bash
uv run mypy .
```

### Unit Tests

```bash
uv run pytest
```

---

## Verwendung von DVC

Falls du reproduzierbare Pipelines und versionierte Artefakte nutzt:

### Pipeline ausführen

```bash
dvc repro
```

### Einzelnen Pipeline-Schritt ausführen

```bash
dvc repro chunk
```

### Dokumente herunterladen (z. B. aus S3)

```bash
dvc pull
```

### Dokumente hochladen

```bash
dvc push
```

### Dokumente zur Versionskontrolle hinzufügen

```bash
dvc add data/raw/documents/sample.pdf
```

## uv

### Pakete hinzufügen

```bash
uv add <paketname>
```

### Lokale Umgebung aktualisieren

```bash
uv sync
```

### Lokfile aktualisieren

```bash
uv lock
```
