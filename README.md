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

4. Kopiere das Template und erstelle deine lokale Config:

   ```bash
   cp env-template env
   ```

5. Fülle `AWS_ACCESS_KEY_ID` und `AWS_SECRET_ACCESS_KEY` aus Schritt 1 in `env` ein und ersetze `[DB_PASSWORD]`.

   Hinweis: Die Datei `env` wird **nicht** versioniert.

### Dokumente herunterladen

```bash
dvc pull
```

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

## MLFlow

### Docker Daemon (bzw. Docker Desktop) vorbereiten
Der Free Tier von Supabase verwendet IPv6 only Adressen. Docker ist by default nicht gerade auf IPv6 optimiert.
Deshalb müsst muss die Daemon Konfiguration hierfür angepasst werden.

Auf Linux ändern oder hinzufügen: /etc/docker/daemon.json
```bash
{
  "ipv6": true,
  "experimental": true,
  "fixed-cidr-v6": "fd00::/80",
  "ip6tables": true,
}
```

Auf Docker Desktop (Mac oder Windows) Setting => Docker Engine editieren. Beispiel:
```bash
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": true,
  "fixed-cidr-v6": "fd00::/80",
  "ip6tables": true,
  "ipv6": true
}
```

HINWEIS: Euer Host muss natürlich auch eine IPv6 Adresse haben. Am besten kurz überprüfen vor dem Start.

### .env File vorbereiten
Folgende ENV Variabeln müssen in einem .env File vorhanden sein.
```bash
MLFLOW_BACKEND_STORE_URI=postgresql://postgres:<INSERT PW PLEASE>@db.hjijyloqvddflojzrvcn.supabase.co:5432/postgres
MLFLOW_DEFAULT_ARTIFACT_ROOT=s3://Imara-mlflow/
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5002

AWS_ACCESS_KEY_ID=<INSERT KEY PLEASE>
AWS_SECRET_ACCESS_KEY=<INSERT KEY PLEASE>
AWS_DEFAULT_REGION=eu-north-1
MLFLOW_S3_ENDPOINT_URL=https://hjijyloqvddflojzrvcn.storage.supabase.co/storage/v1/s3
```

### MLFlow starten
Um MlFlow zu starten einfach folgenden Befehl ausführen und auf localhost mit konfiguriertem Port aufrufen.
```bash
docker compose up -d
```
