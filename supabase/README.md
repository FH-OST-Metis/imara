# Supabase Local Development Environment

Supabase provides a local development environment that allows developers to easily manage databases, authentication, and storage. The local setup enables testing and development of backend features without connecting to external servers. With the Supabase CLI, projects can be quickly initialized and configured. Changes can be tested locally and later deployed to production, supporting an efficient workflow for modern web applications.

This setup includes all core features, such as authentication, databases, and storage buckets. Everything runs inside Docker containers, so Docker must be installed on your system.

## Serving Project local

After you have set up local Supabase ENV (Install and Link). **See install instructions below**

> **Notice:** Docker has to be installed and up and running **

> **Notice:** For Windows: Supabase CLI is installed by Scoop: https://scoop.sh. Don't forget to reboot Windows.**

> **Notice:** For Windows: Has to be executed in PowerShell **

```bash
supabase start
supabase functions serve embed --env-file .env --no-verify-jwt
```

## Important Supabase CLI Commands

The Supabase CLI offers essential commands for managing local and remote development environments. Here are the key commands for environment and migration management:

### Install Supabase CLI

https://supabase.com/docs/guides/local-development/cli/getting-started?queryGroups=platform&platform=macos


### Initialization and Project Linking

> **Notice:** This step has already been completed and committed. **Do not run it again.**

```bash
supabase init # switched to beta, due a bug on the stable version of the cli
```
Initializes a new Supabase project in the current directory.

> **Notice:** The linking has to be done on each local device. **Do this with first time connecting new local ENV **

```bash
supabase login # switched to beta, due a bug on the stable version of the cli
supabase link --project-ref <PROJECT_ID> # switched to beta, due a bug on the stable version of the cli
```
Links the local repository to an existing Supabase project.

### Synchronize Schema Changes

```bash
supabase db pull # switched to beta, due a bug on the stable version of the cli
```
Pulls schema changes from the dashboard and creates a new migration.

### Start Local Development

```bash
supabase start # switched to beta, due a bug on the stable version of the cli
```
Starts the local Supabase environment (Docker required).

### Create and Apply Migrations

**Manual migration:**
```bash
supabase migration new <migration_name> # switched to beta, due a bug on the stable version of the cli
```
Creates a new empty migration file.

**Migration from SQL script:**
```bash
supabase migration new <migration_name> < script.sql # switched to beta, due a bug on the stable version of the cli
```
Creates a migration directly from an existing SQL script.

**Apply migrations:**
```bash
supabase db reset # switched to beta, due a bug on the stable version of the cli
```
Resets the local database and applies all migrations.

### Automatic Schema Diff Migration

```bash
supabase db diff -f <migration_name> # switched to beta, due a bug on the stable version of the cli
```
Generates a migration based on changes in the local database.

---

These commands enable efficient management of development, staging, and production environments with Supabase and GitHub Actions.

## Troubelshoot Windows

If you see this Error:

```bash
failed to start docker container: Error response from daemon: ports are not available: exposing port TCP 0.0.0.0:54322 -> 127.0.0.1:0: listen tcp 0.0.0.0:54322: bind: An attempt was made to access a socket in a way forbidden by its access permissions.
Try rerunning the command with --debug to troubleshoot the error.
```

do execut this in command in PowerShell with Admin Permissions:

```bash
net stop winnat
docker start container_name
net start winnat
```