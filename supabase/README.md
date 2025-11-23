# Supabase Local Development Environment

Supabase provides a local development environment that allows developers to easily manage databases, authentication, and storage. The local setup enables testing and development of backend features without connecting to external servers. With the Supabase CLI, projects can be quickly initialized and configured. Changes can be tested locally and later deployed to production, supporting an efficient workflow for modern web applications.

This setup includes all core features, such as authentication, databases, and storage buckets. Everything runs inside Docker containers, so Docker must be installed on your system.

## Important Supabase CLI Commands

The Supabase CLI offers essential commands for managing local and remote development environments. Here are the key commands for environment and migration management:

### Initialization and Project Linking

> **Notice:** This step has already been completed and committed. **Do not run it again.**

```bash
supabase init
```
Initializes a new Supabase project in the current directory.

> **Notice:** This step couldn't be completed due to a Supabase maintenance window. It will be done later.

```bash
supabase login
supabase link --project-ref <PROJECT_ID>
```
Links the local repository to an existing Supabase project.

### Synchronize Schema Changes

```bash
supabase db pull
```
Pulls schema changes from the dashboard and creates a new migration.

### Start Local Development

```bash
supabase start
```
Starts the local Supabase environment (Docker required).

### Create and Apply Migrations

**Manual migration:**
```bash
supabase migration new <migration_name>
```
Creates a new empty migration file.

**Migration from SQL script:**
```bash
supabase migration new <migration_name> < script.sql
```
Creates a migration directly from an existing SQL script.

**Apply migrations:**
```bash
supabase db reset
```
Resets the local database and applies all migrations.

### Automatic Schema Diff Migration

```bash
supabase db diff -f <migration_name>
```
Generates a migration based on changes in the local database.

---

These commands enable efficient management of development, staging, and production environments with Supabase and GitHub Actions.