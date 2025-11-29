drop extension if exists "pg_net";

alter table "public"."experiments" drop constraint "experiments_lifecycle_stage";

alter table "public"."logged_models" drop constraint "logged_models_lifecycle_stage_check";

alter table "public"."runs" drop constraint "runs_lifecycle_stage";

alter table "public"."runs" drop constraint "runs_status_check";

alter table "public"."runs" drop constraint "source_type";

alter table "public"."experiments" add constraint "experiments_lifecycle_stage" CHECK (((lifecycle_stage)::text = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::text[]))) not valid;

alter table "public"."experiments" validate constraint "experiments_lifecycle_stage";

alter table "public"."logged_models" add constraint "logged_models_lifecycle_stage_check" CHECK (((lifecycle_stage)::text = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::text[]))) not valid;

alter table "public"."logged_models" validate constraint "logged_models_lifecycle_stage_check";

alter table "public"."runs" add constraint "runs_lifecycle_stage" CHECK (((lifecycle_stage)::text = ANY ((ARRAY['active'::character varying, 'deleted'::character varying])::text[]))) not valid;

alter table "public"."runs" validate constraint "runs_lifecycle_stage";

alter table "public"."runs" add constraint "runs_status_check" CHECK (((status)::text = ANY ((ARRAY['SCHEDULED'::character varying, 'FAILED'::character varying, 'FINISHED'::character varying, 'RUNNING'::character varying, 'KILLED'::character varying])::text[]))) not valid;

alter table "public"."runs" validate constraint "runs_status_check";

alter table "public"."runs" add constraint "source_type" CHECK (((source_type)::text = ANY ((ARRAY['NOTEBOOK'::character varying, 'JOB'::character varying, 'LOCAL'::character varying, 'UNKNOWN'::character varying, 'PROJECT'::character varying])::text[]))) not valid;

alter table "public"."runs" validate constraint "source_type";


