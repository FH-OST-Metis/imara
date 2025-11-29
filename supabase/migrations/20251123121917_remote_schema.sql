


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";





SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."alembic_version" (
    "version_num" character varying(32) NOT NULL
);


ALTER TABLE "public"."alembic_version" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."assessments" (
    "assessment_id" character varying(50) NOT NULL,
    "trace_id" character varying(50) NOT NULL,
    "name" character varying(250) NOT NULL,
    "assessment_type" character varying(20) NOT NULL,
    "value" "text" NOT NULL,
    "error" "text",
    "created_timestamp" bigint NOT NULL,
    "last_updated_timestamp" bigint NOT NULL,
    "source_type" character varying(50) NOT NULL,
    "source_id" character varying(250),
    "run_id" character varying(32),
    "span_id" character varying(50),
    "rationale" "text",
    "overrides" character varying(50),
    "valid" boolean NOT NULL,
    "assessment_metadata" "text"
);


ALTER TABLE "public"."assessments" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."datasets" (
    "dataset_uuid" character varying(36) NOT NULL,
    "experiment_id" integer NOT NULL,
    "name" character varying(500) NOT NULL,
    "digest" character varying(36) NOT NULL,
    "dataset_source_type" character varying(36) NOT NULL,
    "dataset_source" "text" NOT NULL,
    "dataset_schema" "text",
    "dataset_profile" "text"
);


ALTER TABLE "public"."datasets" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."entity_associations" (
    "association_id" character varying(36) NOT NULL,
    "source_type" character varying(36) NOT NULL,
    "source_id" character varying(36) NOT NULL,
    "destination_type" character varying(36) NOT NULL,
    "destination_id" character varying(36) NOT NULL,
    "created_time" bigint
);


ALTER TABLE "public"."entity_associations" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."evaluation_dataset_records" (
    "dataset_record_id" character varying(36) NOT NULL,
    "dataset_id" character varying(36) NOT NULL,
    "inputs" json NOT NULL,
    "expectations" json,
    "tags" json,
    "source" json,
    "source_id" character varying(36),
    "source_type" character varying(255),
    "created_time" bigint,
    "last_update_time" bigint,
    "created_by" character varying(255),
    "last_updated_by" character varying(255),
    "input_hash" character varying(64) NOT NULL,
    "outputs" json
);


ALTER TABLE "public"."evaluation_dataset_records" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."evaluation_dataset_tags" (
    "dataset_id" character varying(36) NOT NULL,
    "key" character varying(255) NOT NULL,
    "value" character varying(5000)
);


ALTER TABLE "public"."evaluation_dataset_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."evaluation_datasets" (
    "dataset_id" character varying(36) NOT NULL,
    "name" character varying(255) NOT NULL,
    "schema" "text",
    "profile" "text",
    "digest" character varying(64),
    "created_time" bigint,
    "last_update_time" bigint,
    "created_by" character varying(255),
    "last_updated_by" character varying(255)
);


ALTER TABLE "public"."evaluation_datasets" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."experiment_tags" (
    "key" character varying(250) NOT NULL,
    "value" character varying(5000),
    "experiment_id" integer NOT NULL
);


ALTER TABLE "public"."experiment_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."experiments" (
    "experiment_id" integer NOT NULL,
    "name" character varying(256) NOT NULL,
    "artifact_location" character varying(256),
    "lifecycle_stage" character varying(32),
    "creation_time" bigint,
    "last_update_time" bigint,
    CONSTRAINT "experiments_lifecycle_stage" CHECK ((("lifecycle_stage")::"text" = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::"text"[])))
);


ALTER TABLE "public"."experiments" OWNER TO "postgres";


CREATE SEQUENCE IF NOT EXISTS "public"."experiments_experiment_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER SEQUENCE "public"."experiments_experiment_id_seq" OWNER TO "postgres";


ALTER SEQUENCE "public"."experiments_experiment_id_seq" OWNED BY "public"."experiments"."experiment_id";



CREATE TABLE IF NOT EXISTS "public"."input_tags" (
    "input_uuid" character varying(36) NOT NULL,
    "name" character varying(255) NOT NULL,
    "value" character varying(500) NOT NULL
);


ALTER TABLE "public"."input_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."inputs" (
    "input_uuid" character varying(36) NOT NULL,
    "source_type" character varying(36) NOT NULL,
    "source_id" character varying(36) NOT NULL,
    "destination_type" character varying(36) NOT NULL,
    "destination_id" character varying(36) NOT NULL,
    "step" bigint DEFAULT '0'::bigint NOT NULL
);


ALTER TABLE "public"."inputs" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."jobs" (
    "id" character varying(36) NOT NULL,
    "creation_time" bigint NOT NULL,
    "function_fullname" character varying(500) NOT NULL,
    "params" "text" NOT NULL,
    "timeout" double precision,
    "status" integer NOT NULL,
    "result" "text",
    "retry_count" integer NOT NULL,
    "last_update_time" bigint NOT NULL
);


ALTER TABLE "public"."jobs" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."latest_metrics" (
    "key" character varying(250) NOT NULL,
    "value" double precision NOT NULL,
    "timestamp" bigint,
    "step" bigint NOT NULL,
    "is_nan" boolean NOT NULL,
    "run_uuid" character varying(32) NOT NULL
);


ALTER TABLE "public"."latest_metrics" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."logged_model_metrics" (
    "model_id" character varying(36) NOT NULL,
    "metric_name" character varying(500) NOT NULL,
    "metric_timestamp_ms" bigint NOT NULL,
    "metric_step" bigint NOT NULL,
    "metric_value" double precision,
    "experiment_id" integer NOT NULL,
    "run_id" character varying(32) NOT NULL,
    "dataset_uuid" character varying(36),
    "dataset_name" character varying(500),
    "dataset_digest" character varying(36)
);


ALTER TABLE "public"."logged_model_metrics" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."logged_model_params" (
    "model_id" character varying(36) NOT NULL,
    "experiment_id" integer NOT NULL,
    "param_key" character varying(255) NOT NULL,
    "param_value" "text" NOT NULL
);


ALTER TABLE "public"."logged_model_params" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."logged_model_tags" (
    "model_id" character varying(36) NOT NULL,
    "experiment_id" integer NOT NULL,
    "tag_key" character varying(255) NOT NULL,
    "tag_value" "text" NOT NULL
);


ALTER TABLE "public"."logged_model_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."logged_models" (
    "model_id" character varying(36) NOT NULL,
    "experiment_id" integer NOT NULL,
    "name" character varying(500) NOT NULL,
    "artifact_location" character varying(1000) NOT NULL,
    "creation_timestamp_ms" bigint NOT NULL,
    "last_updated_timestamp_ms" bigint NOT NULL,
    "status" integer NOT NULL,
    "lifecycle_stage" character varying(32),
    "model_type" character varying(500),
    "source_run_id" character varying(32),
    "status_message" character varying(1000),
    CONSTRAINT "logged_models_lifecycle_stage_check" CHECK ((("lifecycle_stage")::"text" = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::"text"[])))
);


ALTER TABLE "public"."logged_models" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."metrics" (
    "key" character varying(250) NOT NULL,
    "value" double precision NOT NULL,
    "timestamp" bigint NOT NULL,
    "run_uuid" character varying(32) NOT NULL,
    "step" bigint DEFAULT '0'::bigint NOT NULL,
    "is_nan" boolean DEFAULT false NOT NULL
);


ALTER TABLE "public"."metrics" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."model_version_tags" (
    "key" character varying(250) NOT NULL,
    "value" "text",
    "name" character varying(256) NOT NULL,
    "version" integer NOT NULL
);


ALTER TABLE "public"."model_version_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."model_versions" (
    "name" character varying(256) NOT NULL,
    "version" integer NOT NULL,
    "creation_time" bigint,
    "last_updated_time" bigint,
    "description" character varying(5000),
    "user_id" character varying(256),
    "current_stage" character varying(20),
    "source" character varying(500),
    "run_id" character varying(32),
    "status" character varying(20),
    "status_message" character varying(500),
    "run_link" character varying(500),
    "storage_location" character varying(500)
);


ALTER TABLE "public"."model_versions" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."params" (
    "key" character varying(250) NOT NULL,
    "value" character varying(8000) NOT NULL,
    "run_uuid" character varying(32) NOT NULL
);


ALTER TABLE "public"."params" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."registered_model_aliases" (
    "alias" character varying(256) NOT NULL,
    "version" integer NOT NULL,
    "name" character varying(256) NOT NULL
);


ALTER TABLE "public"."registered_model_aliases" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."registered_model_tags" (
    "key" character varying(250) NOT NULL,
    "value" character varying(5000),
    "name" character varying(256) NOT NULL
);


ALTER TABLE "public"."registered_model_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."registered_models" (
    "name" character varying(256) NOT NULL,
    "creation_time" bigint,
    "last_updated_time" bigint,
    "description" character varying(5000)
);


ALTER TABLE "public"."registered_models" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."runs" (
    "run_uuid" character varying(32) NOT NULL,
    "name" character varying(250),
    "source_type" character varying(20),
    "source_name" character varying(500),
    "entry_point_name" character varying(50),
    "user_id" character varying(256),
    "status" character varying(9),
    "start_time" bigint,
    "end_time" bigint,
    "source_version" character varying(50),
    "lifecycle_stage" character varying(20),
    "artifact_uri" character varying(200),
    "experiment_id" integer,
    "deleted_time" bigint,
    CONSTRAINT "runs_lifecycle_stage" CHECK ((("lifecycle_stage")::"text" = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::"text"[]))),
    CONSTRAINT "runs_status_check" CHECK ((("status")::"text" = ANY ((ARRAY['SCHEDULED'::character varying, 'FAILED'::character varying, 'FINISHED'::character varying, 'RUNNING'::character varying, 'KILLED'::character varying])::"text"[]))),
    CONSTRAINT "source_type" CHECK ((("source_type")::"text" = ANY ((ARRAY['NOTEBOOK'::character varying, 'JOB'::character varying, 'LOCAL'::character varying, 'UNKNOWN'::character varying, 'PROJECT'::character varying])::"text"[])))
);


ALTER TABLE "public"."runs" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."scorer_versions" (
    "scorer_id" character varying(36) NOT NULL,
    "scorer_version" integer NOT NULL,
    "serialized_scorer" "text" NOT NULL,
    "creation_time" bigint
);


ALTER TABLE "public"."scorer_versions" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."scorers" (
    "experiment_id" integer NOT NULL,
    "scorer_name" character varying(256) NOT NULL,
    "scorer_id" character varying(36) NOT NULL
);


ALTER TABLE "public"."scorers" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."spans" (
    "trace_id" character varying(50) NOT NULL,
    "experiment_id" integer NOT NULL,
    "span_id" character varying(50) NOT NULL,
    "parent_span_id" character varying(50),
    "name" "text",
    "type" character varying(500),
    "status" character varying(50) NOT NULL,
    "start_time_unix_nano" bigint NOT NULL,
    "end_time_unix_nano" bigint,
    "duration_ns" bigint GENERATED ALWAYS AS (("end_time_unix_nano" - "start_time_unix_nano")) STORED,
    "content" "text" NOT NULL
);


ALTER TABLE "public"."spans" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."tags" (
    "key" character varying(250) NOT NULL,
    "value" character varying(8000),
    "run_uuid" character varying(32) NOT NULL
);


ALTER TABLE "public"."tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."trace_info" (
    "request_id" character varying(50) NOT NULL,
    "experiment_id" integer NOT NULL,
    "timestamp_ms" bigint NOT NULL,
    "execution_time_ms" bigint,
    "status" character varying(50) NOT NULL,
    "client_request_id" character varying(50),
    "request_preview" character varying(1000),
    "response_preview" character varying(1000)
);


ALTER TABLE "public"."trace_info" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."trace_request_metadata" (
    "key" character varying(250) NOT NULL,
    "value" character varying(8000),
    "request_id" character varying(50) NOT NULL
);


ALTER TABLE "public"."trace_request_metadata" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."trace_tags" (
    "key" character varying(250) NOT NULL,
    "value" character varying(8000),
    "request_id" character varying(50) NOT NULL
);


ALTER TABLE "public"."trace_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."webhook_events" (
    "webhook_id" character varying(256) NOT NULL,
    "entity" character varying(50) NOT NULL,
    "action" character varying(50) NOT NULL
);


ALTER TABLE "public"."webhook_events" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."webhooks" (
    "webhook_id" character varying(256) NOT NULL,
    "name" character varying(256) NOT NULL,
    "description" character varying(1000),
    "url" character varying(500) NOT NULL,
    "status" character varying(20) DEFAULT 'ACTIVE'::character varying NOT NULL,
    "secret" character varying(1000),
    "creation_timestamp" bigint,
    "last_updated_timestamp" bigint,
    "deleted_timestamp" bigint
);


ALTER TABLE "public"."webhooks" OWNER TO "postgres";


ALTER TABLE ONLY "public"."experiments" ALTER COLUMN "experiment_id" SET DEFAULT "nextval"('"public"."experiments_experiment_id_seq"'::"regclass");



ALTER TABLE ONLY "public"."alembic_version"
    ADD CONSTRAINT "alembic_version_pkc" PRIMARY KEY ("version_num");



ALTER TABLE ONLY "public"."assessments"
    ADD CONSTRAINT "assessments_pk" PRIMARY KEY ("assessment_id");



ALTER TABLE ONLY "public"."datasets"
    ADD CONSTRAINT "dataset_pk" PRIMARY KEY ("experiment_id", "name", "digest");



ALTER TABLE ONLY "public"."entity_associations"
    ADD CONSTRAINT "entity_associations_pk" PRIMARY KEY ("source_type", "source_id", "destination_type", "destination_id");



ALTER TABLE ONLY "public"."evaluation_dataset_records"
    ADD CONSTRAINT "evaluation_dataset_records_pk" PRIMARY KEY ("dataset_record_id");



ALTER TABLE ONLY "public"."evaluation_dataset_tags"
    ADD CONSTRAINT "evaluation_dataset_tags_pk" PRIMARY KEY ("dataset_id", "key");



ALTER TABLE ONLY "public"."evaluation_datasets"
    ADD CONSTRAINT "evaluation_datasets_pk" PRIMARY KEY ("dataset_id");



ALTER TABLE ONLY "public"."experiments"
    ADD CONSTRAINT "experiment_pk" PRIMARY KEY ("experiment_id");



ALTER TABLE ONLY "public"."experiment_tags"
    ADD CONSTRAINT "experiment_tag_pk" PRIMARY KEY ("key", "experiment_id");



ALTER TABLE ONLY "public"."experiments"
    ADD CONSTRAINT "experiments_name_key" UNIQUE ("name");



ALTER TABLE ONLY "public"."input_tags"
    ADD CONSTRAINT "input_tags_pk" PRIMARY KEY ("input_uuid", "name");



ALTER TABLE ONLY "public"."inputs"
    ADD CONSTRAINT "inputs_pk" PRIMARY KEY ("source_type", "source_id", "destination_type", "destination_id");



ALTER TABLE ONLY "public"."jobs"
    ADD CONSTRAINT "jobs_pk" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."latest_metrics"
    ADD CONSTRAINT "latest_metric_pk" PRIMARY KEY ("key", "run_uuid");



ALTER TABLE ONLY "public"."logged_model_metrics"
    ADD CONSTRAINT "logged_model_metrics_pk" PRIMARY KEY ("model_id", "metric_name", "metric_timestamp_ms", "metric_step", "run_id");



ALTER TABLE ONLY "public"."logged_model_params"
    ADD CONSTRAINT "logged_model_params_pk" PRIMARY KEY ("model_id", "param_key");



ALTER TABLE ONLY "public"."logged_model_tags"
    ADD CONSTRAINT "logged_model_tags_pk" PRIMARY KEY ("model_id", "tag_key");



ALTER TABLE ONLY "public"."logged_models"
    ADD CONSTRAINT "logged_models_pk" PRIMARY KEY ("model_id");



ALTER TABLE ONLY "public"."metrics"
    ADD CONSTRAINT "metric_pk" PRIMARY KEY ("key", "timestamp", "step", "run_uuid", "value", "is_nan");



ALTER TABLE ONLY "public"."model_versions"
    ADD CONSTRAINT "model_version_pk" PRIMARY KEY ("name", "version");



ALTER TABLE ONLY "public"."model_version_tags"
    ADD CONSTRAINT "model_version_tag_pk" PRIMARY KEY ("key", "name", "version");



ALTER TABLE ONLY "public"."params"
    ADD CONSTRAINT "param_pk" PRIMARY KEY ("key", "run_uuid");



ALTER TABLE ONLY "public"."registered_model_aliases"
    ADD CONSTRAINT "registered_model_alias_pk" PRIMARY KEY ("name", "alias");



ALTER TABLE ONLY "public"."registered_models"
    ADD CONSTRAINT "registered_model_pk" PRIMARY KEY ("name");



ALTER TABLE ONLY "public"."registered_model_tags"
    ADD CONSTRAINT "registered_model_tag_pk" PRIMARY KEY ("key", "name");



ALTER TABLE ONLY "public"."runs"
    ADD CONSTRAINT "run_pk" PRIMARY KEY ("run_uuid");



ALTER TABLE ONLY "public"."scorers"
    ADD CONSTRAINT "scorer_pk" PRIMARY KEY ("scorer_id");



ALTER TABLE ONLY "public"."scorer_versions"
    ADD CONSTRAINT "scorer_version_pk" PRIMARY KEY ("scorer_id", "scorer_version");



ALTER TABLE ONLY "public"."spans"
    ADD CONSTRAINT "spans_pk" PRIMARY KEY ("trace_id", "span_id");



ALTER TABLE ONLY "public"."tags"
    ADD CONSTRAINT "tag_pk" PRIMARY KEY ("key", "run_uuid");



ALTER TABLE ONLY "public"."trace_info"
    ADD CONSTRAINT "trace_info_pk" PRIMARY KEY ("request_id");



ALTER TABLE ONLY "public"."trace_request_metadata"
    ADD CONSTRAINT "trace_request_metadata_pk" PRIMARY KEY ("key", "request_id");



ALTER TABLE ONLY "public"."trace_tags"
    ADD CONSTRAINT "trace_tag_pk" PRIMARY KEY ("key", "request_id");



ALTER TABLE ONLY "public"."evaluation_dataset_records"
    ADD CONSTRAINT "unique_dataset_input" UNIQUE ("dataset_id", "input_hash");



ALTER TABLE ONLY "public"."webhook_events"
    ADD CONSTRAINT "webhook_event_pk" PRIMARY KEY ("webhook_id", "entity", "action");



ALTER TABLE ONLY "public"."webhooks"
    ADD CONSTRAINT "webhook_pk" PRIMARY KEY ("webhook_id");



CREATE INDEX "idx_webhook_events_action" ON "public"."webhook_events" USING "btree" ("action");



CREATE INDEX "idx_webhook_events_entity" ON "public"."webhook_events" USING "btree" ("entity");



CREATE INDEX "idx_webhook_events_entity_action" ON "public"."webhook_events" USING "btree" ("entity", "action");



CREATE INDEX "idx_webhooks_name" ON "public"."webhooks" USING "btree" ("name");



CREATE INDEX "idx_webhooks_status" ON "public"."webhooks" USING "btree" ("status");



CREATE INDEX "index_assessments_assessment_type" ON "public"."assessments" USING "btree" ("assessment_type");



CREATE INDEX "index_assessments_last_updated_timestamp" ON "public"."assessments" USING "btree" ("last_updated_timestamp");



CREATE INDEX "index_assessments_run_id_created_timestamp" ON "public"."assessments" USING "btree" ("run_id", "created_timestamp");



CREATE INDEX "index_assessments_trace_id_created_timestamp" ON "public"."assessments" USING "btree" ("trace_id", "created_timestamp");



CREATE INDEX "index_datasets_dataset_uuid" ON "public"."datasets" USING "btree" ("dataset_uuid");



CREATE INDEX "index_datasets_experiment_id_dataset_source_type" ON "public"."datasets" USING "btree" ("experiment_id", "dataset_source_type");



CREATE INDEX "index_entity_associations_association_id" ON "public"."entity_associations" USING "btree" ("association_id");



CREATE INDEX "index_entity_associations_reverse_lookup" ON "public"."entity_associations" USING "btree" ("destination_type", "destination_id", "source_type", "source_id");



CREATE INDEX "index_evaluation_dataset_records_dataset_id" ON "public"."evaluation_dataset_records" USING "btree" ("dataset_id");



CREATE INDEX "index_evaluation_dataset_tags_dataset_id" ON "public"."evaluation_dataset_tags" USING "btree" ("dataset_id");



CREATE INDEX "index_evaluation_datasets_created_time" ON "public"."evaluation_datasets" USING "btree" ("created_time");



CREATE INDEX "index_evaluation_datasets_name" ON "public"."evaluation_datasets" USING "btree" ("name");



CREATE INDEX "index_inputs_destination_type_destination_id_source_type" ON "public"."inputs" USING "btree" ("destination_type", "destination_id", "source_type");



CREATE INDEX "index_inputs_input_uuid" ON "public"."inputs" USING "btree" ("input_uuid");



CREATE INDEX "index_jobs_function_status_creation_time" ON "public"."jobs" USING "btree" ("function_fullname", "status", "creation_time");



CREATE INDEX "index_latest_metrics_run_uuid" ON "public"."latest_metrics" USING "btree" ("run_uuid");



CREATE INDEX "index_logged_model_metrics_model_id" ON "public"."logged_model_metrics" USING "btree" ("model_id");



CREATE INDEX "index_metrics_run_uuid" ON "public"."metrics" USING "btree" ("run_uuid");



CREATE INDEX "index_params_run_uuid" ON "public"."params" USING "btree" ("run_uuid");



CREATE INDEX "index_scorer_versions_scorer_id" ON "public"."scorer_versions" USING "btree" ("scorer_id");



CREATE UNIQUE INDEX "index_scorers_experiment_id_scorer_name" ON "public"."scorers" USING "btree" ("experiment_id", "scorer_name");



CREATE INDEX "index_spans_experiment_id" ON "public"."spans" USING "btree" ("experiment_id");



CREATE INDEX "index_spans_experiment_id_duration" ON "public"."spans" USING "btree" ("experiment_id", "duration_ns");



CREATE INDEX "index_spans_experiment_id_status_type" ON "public"."spans" USING "btree" ("experiment_id", "status", "type");



CREATE INDEX "index_spans_experiment_id_type_status" ON "public"."spans" USING "btree" ("experiment_id", "type", "status");



CREATE INDEX "index_tags_run_uuid" ON "public"."tags" USING "btree" ("run_uuid");



CREATE INDEX "index_trace_info_experiment_id_timestamp_ms" ON "public"."trace_info" USING "btree" ("experiment_id", "timestamp_ms");



CREATE INDEX "index_trace_request_metadata_request_id" ON "public"."trace_request_metadata" USING "btree" ("request_id");



CREATE INDEX "index_trace_tags_request_id" ON "public"."trace_tags" USING "btree" ("request_id");



ALTER TABLE ONLY "public"."experiment_tags"
    ADD CONSTRAINT "experiment_tags_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id");



ALTER TABLE ONLY "public"."assessments"
    ADD CONSTRAINT "fk_assessments_trace_id" FOREIGN KEY ("trace_id") REFERENCES "public"."trace_info"("request_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."datasets"
    ADD CONSTRAINT "fk_datasets_experiment_id_experiments" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."evaluation_dataset_records"
    ADD CONSTRAINT "fk_evaluation_dataset_records_dataset_id" FOREIGN KEY ("dataset_id") REFERENCES "public"."evaluation_datasets"("dataset_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."evaluation_dataset_tags"
    ADD CONSTRAINT "fk_evaluation_dataset_tags_dataset_id" FOREIGN KEY ("dataset_id") REFERENCES "public"."evaluation_datasets"("dataset_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."logged_model_metrics"
    ADD CONSTRAINT "fk_logged_model_metrics_experiment_id" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id");



ALTER TABLE ONLY "public"."logged_model_metrics"
    ADD CONSTRAINT "fk_logged_model_metrics_model_id" FOREIGN KEY ("model_id") REFERENCES "public"."logged_models"("model_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."logged_model_metrics"
    ADD CONSTRAINT "fk_logged_model_metrics_run_id" FOREIGN KEY ("run_id") REFERENCES "public"."runs"("run_uuid") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."logged_model_params"
    ADD CONSTRAINT "fk_logged_model_params_experiment_id" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id");



ALTER TABLE ONLY "public"."logged_model_params"
    ADD CONSTRAINT "fk_logged_model_params_model_id" FOREIGN KEY ("model_id") REFERENCES "public"."logged_models"("model_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."logged_model_tags"
    ADD CONSTRAINT "fk_logged_model_tags_experiment_id" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id");



ALTER TABLE ONLY "public"."logged_model_tags"
    ADD CONSTRAINT "fk_logged_model_tags_model_id" FOREIGN KEY ("model_id") REFERENCES "public"."logged_models"("model_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."logged_models"
    ADD CONSTRAINT "fk_logged_models_experiment_id" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."scorer_versions"
    ADD CONSTRAINT "fk_scorer_versions_scorer_id" FOREIGN KEY ("scorer_id") REFERENCES "public"."scorers"("scorer_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."scorers"
    ADD CONSTRAINT "fk_scorers_experiment_id" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."spans"
    ADD CONSTRAINT "fk_spans_experiment_id" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id");



ALTER TABLE ONLY "public"."spans"
    ADD CONSTRAINT "fk_spans_trace_id" FOREIGN KEY ("trace_id") REFERENCES "public"."trace_info"("request_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."trace_info"
    ADD CONSTRAINT "fk_trace_info_experiment_id" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id");



ALTER TABLE ONLY "public"."trace_request_metadata"
    ADD CONSTRAINT "fk_trace_request_metadata_request_id" FOREIGN KEY ("request_id") REFERENCES "public"."trace_info"("request_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."trace_tags"
    ADD CONSTRAINT "fk_trace_tags_request_id" FOREIGN KEY ("request_id") REFERENCES "public"."trace_info"("request_id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."latest_metrics"
    ADD CONSTRAINT "latest_metrics_run_uuid_fkey" FOREIGN KEY ("run_uuid") REFERENCES "public"."runs"("run_uuid");



ALTER TABLE ONLY "public"."metrics"
    ADD CONSTRAINT "metrics_run_uuid_fkey" FOREIGN KEY ("run_uuid") REFERENCES "public"."runs"("run_uuid");



ALTER TABLE ONLY "public"."model_version_tags"
    ADD CONSTRAINT "model_version_tags_name_version_fkey" FOREIGN KEY ("name", "version") REFERENCES "public"."model_versions"("name", "version") ON UPDATE CASCADE;



ALTER TABLE ONLY "public"."model_versions"
    ADD CONSTRAINT "model_versions_name_fkey" FOREIGN KEY ("name") REFERENCES "public"."registered_models"("name") ON UPDATE CASCADE;



ALTER TABLE ONLY "public"."params"
    ADD CONSTRAINT "params_run_uuid_fkey" FOREIGN KEY ("run_uuid") REFERENCES "public"."runs"("run_uuid");



ALTER TABLE ONLY "public"."registered_model_aliases"
    ADD CONSTRAINT "registered_model_alias_name_fkey" FOREIGN KEY ("name") REFERENCES "public"."registered_models"("name") ON UPDATE CASCADE ON DELETE CASCADE;



ALTER TABLE ONLY "public"."registered_model_tags"
    ADD CONSTRAINT "registered_model_tags_name_fkey" FOREIGN KEY ("name") REFERENCES "public"."registered_models"("name") ON UPDATE CASCADE;



ALTER TABLE ONLY "public"."runs"
    ADD CONSTRAINT "runs_experiment_id_fkey" FOREIGN KEY ("experiment_id") REFERENCES "public"."experiments"("experiment_id");



ALTER TABLE ONLY "public"."tags"
    ADD CONSTRAINT "tags_run_uuid_fkey" FOREIGN KEY ("run_uuid") REFERENCES "public"."runs"("run_uuid");



ALTER TABLE ONLY "public"."webhook_events"
    ADD CONSTRAINT "webhook_events_webhook_id_fkey" FOREIGN KEY ("webhook_id") REFERENCES "public"."webhooks"("webhook_id") ON DELETE CASCADE;





ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";


GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";








































































































































































GRANT ALL ON TABLE "public"."alembic_version" TO "anon";
GRANT ALL ON TABLE "public"."alembic_version" TO "authenticated";
GRANT ALL ON TABLE "public"."alembic_version" TO "service_role";



GRANT ALL ON TABLE "public"."assessments" TO "anon";
GRANT ALL ON TABLE "public"."assessments" TO "authenticated";
GRANT ALL ON TABLE "public"."assessments" TO "service_role";



GRANT ALL ON TABLE "public"."datasets" TO "anon";
GRANT ALL ON TABLE "public"."datasets" TO "authenticated";
GRANT ALL ON TABLE "public"."datasets" TO "service_role";



GRANT ALL ON TABLE "public"."entity_associations" TO "anon";
GRANT ALL ON TABLE "public"."entity_associations" TO "authenticated";
GRANT ALL ON TABLE "public"."entity_associations" TO "service_role";



GRANT ALL ON TABLE "public"."evaluation_dataset_records" TO "anon";
GRANT ALL ON TABLE "public"."evaluation_dataset_records" TO "authenticated";
GRANT ALL ON TABLE "public"."evaluation_dataset_records" TO "service_role";



GRANT ALL ON TABLE "public"."evaluation_dataset_tags" TO "anon";
GRANT ALL ON TABLE "public"."evaluation_dataset_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."evaluation_dataset_tags" TO "service_role";



GRANT ALL ON TABLE "public"."evaluation_datasets" TO "anon";
GRANT ALL ON TABLE "public"."evaluation_datasets" TO "authenticated";
GRANT ALL ON TABLE "public"."evaluation_datasets" TO "service_role";



GRANT ALL ON TABLE "public"."experiment_tags" TO "anon";
GRANT ALL ON TABLE "public"."experiment_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."experiment_tags" TO "service_role";



GRANT ALL ON TABLE "public"."experiments" TO "anon";
GRANT ALL ON TABLE "public"."experiments" TO "authenticated";
GRANT ALL ON TABLE "public"."experiments" TO "service_role";



GRANT ALL ON SEQUENCE "public"."experiments_experiment_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."experiments_experiment_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."experiments_experiment_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."input_tags" TO "anon";
GRANT ALL ON TABLE "public"."input_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."input_tags" TO "service_role";



GRANT ALL ON TABLE "public"."inputs" TO "anon";
GRANT ALL ON TABLE "public"."inputs" TO "authenticated";
GRANT ALL ON TABLE "public"."inputs" TO "service_role";



GRANT ALL ON TABLE "public"."jobs" TO "anon";
GRANT ALL ON TABLE "public"."jobs" TO "authenticated";
GRANT ALL ON TABLE "public"."jobs" TO "service_role";



GRANT ALL ON TABLE "public"."latest_metrics" TO "anon";
GRANT ALL ON TABLE "public"."latest_metrics" TO "authenticated";
GRANT ALL ON TABLE "public"."latest_metrics" TO "service_role";



GRANT ALL ON TABLE "public"."logged_model_metrics" TO "anon";
GRANT ALL ON TABLE "public"."logged_model_metrics" TO "authenticated";
GRANT ALL ON TABLE "public"."logged_model_metrics" TO "service_role";



GRANT ALL ON TABLE "public"."logged_model_params" TO "anon";
GRANT ALL ON TABLE "public"."logged_model_params" TO "authenticated";
GRANT ALL ON TABLE "public"."logged_model_params" TO "service_role";



GRANT ALL ON TABLE "public"."logged_model_tags" TO "anon";
GRANT ALL ON TABLE "public"."logged_model_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."logged_model_tags" TO "service_role";



GRANT ALL ON TABLE "public"."logged_models" TO "anon";
GRANT ALL ON TABLE "public"."logged_models" TO "authenticated";
GRANT ALL ON TABLE "public"."logged_models" TO "service_role";



GRANT ALL ON TABLE "public"."metrics" TO "anon";
GRANT ALL ON TABLE "public"."metrics" TO "authenticated";
GRANT ALL ON TABLE "public"."metrics" TO "service_role";



GRANT ALL ON TABLE "public"."model_version_tags" TO "anon";
GRANT ALL ON TABLE "public"."model_version_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."model_version_tags" TO "service_role";



GRANT ALL ON TABLE "public"."model_versions" TO "anon";
GRANT ALL ON TABLE "public"."model_versions" TO "authenticated";
GRANT ALL ON TABLE "public"."model_versions" TO "service_role";



GRANT ALL ON TABLE "public"."params" TO "anon";
GRANT ALL ON TABLE "public"."params" TO "authenticated";
GRANT ALL ON TABLE "public"."params" TO "service_role";



GRANT ALL ON TABLE "public"."registered_model_aliases" TO "anon";
GRANT ALL ON TABLE "public"."registered_model_aliases" TO "authenticated";
GRANT ALL ON TABLE "public"."registered_model_aliases" TO "service_role";



GRANT ALL ON TABLE "public"."registered_model_tags" TO "anon";
GRANT ALL ON TABLE "public"."registered_model_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."registered_model_tags" TO "service_role";



GRANT ALL ON TABLE "public"."registered_models" TO "anon";
GRANT ALL ON TABLE "public"."registered_models" TO "authenticated";
GRANT ALL ON TABLE "public"."registered_models" TO "service_role";



GRANT ALL ON TABLE "public"."runs" TO "anon";
GRANT ALL ON TABLE "public"."runs" TO "authenticated";
GRANT ALL ON TABLE "public"."runs" TO "service_role";



GRANT ALL ON TABLE "public"."scorer_versions" TO "anon";
GRANT ALL ON TABLE "public"."scorer_versions" TO "authenticated";
GRANT ALL ON TABLE "public"."scorer_versions" TO "service_role";



GRANT ALL ON TABLE "public"."scorers" TO "anon";
GRANT ALL ON TABLE "public"."scorers" TO "authenticated";
GRANT ALL ON TABLE "public"."scorers" TO "service_role";



GRANT ALL ON TABLE "public"."spans" TO "anon";
GRANT ALL ON TABLE "public"."spans" TO "authenticated";
GRANT ALL ON TABLE "public"."spans" TO "service_role";



GRANT ALL ON TABLE "public"."tags" TO "anon";
GRANT ALL ON TABLE "public"."tags" TO "authenticated";
GRANT ALL ON TABLE "public"."tags" TO "service_role";



GRANT ALL ON TABLE "public"."trace_info" TO "anon";
GRANT ALL ON TABLE "public"."trace_info" TO "authenticated";
GRANT ALL ON TABLE "public"."trace_info" TO "service_role";



GRANT ALL ON TABLE "public"."trace_request_metadata" TO "anon";
GRANT ALL ON TABLE "public"."trace_request_metadata" TO "authenticated";
GRANT ALL ON TABLE "public"."trace_request_metadata" TO "service_role";



GRANT ALL ON TABLE "public"."trace_tags" TO "anon";
GRANT ALL ON TABLE "public"."trace_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."trace_tags" TO "service_role";



GRANT ALL ON TABLE "public"."webhook_events" TO "anon";
GRANT ALL ON TABLE "public"."webhook_events" TO "authenticated";
GRANT ALL ON TABLE "public"."webhook_events" TO "service_role";



GRANT ALL ON TABLE "public"."webhooks" TO "anon";
GRANT ALL ON TABLE "public"."webhooks" TO "authenticated";
GRANT ALL ON TABLE "public"."webhooks" TO "service_role";









ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "service_role";































