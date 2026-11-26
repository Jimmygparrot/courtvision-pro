-- CourtVision Pro: permanent prediction history (PostgreSQL / Supabase)
-- Run once as the project owner (postgres). Single transaction: all or nothing.
-- Contains NO passwords. Set the app role password separately (see section 6).

BEGIN;

-- 1. Verified, exact pregame predictions ------------------------------------
CREATE TABLE public.predictions (
    game_id              text             PRIMARY KEY,
    game_date            date             NOT NULL,
    tipoff_utc           timestamptz,
    home_team_id         text             NOT NULL,
    home_team            text             NOT NULL,
    away_team_id         text             NOT NULL,
    away_team            text             NOT NULL,
    predicted_winner     text             NOT NULL,
    home_win_probability double precision NOT NULL,
    away_win_probability double precision NOT NULL,
    win_probability      double precision NOT NULL,
    created_at           timestamptz      NOT NULL,
    model_version        text             NOT NULL,
    rating_source        text             NOT NULL,
    actual_winner        text,
    completed_at         timestamptz,
    CONSTRAINT predictions_game_id_format CHECK (game_id ~ '^[0-9]{10}$'),
    CONSTRAINT predictions_distinct_teams CHECK (home_team_id <> away_team_id),
    CONSTRAINT predictions_winner_is_team CHECK (predicted_winner IN (home_team, away_team)),
    CONSTRAINT predictions_prob_range     CHECK (
        home_win_probability BETWEEN 0 AND 1 AND away_win_probability BETWEEN 0 AND 1
        AND win_probability BETWEEN 0.5 AND 1),
    CONSTRAINT predictions_prob_sum       CHECK (abs(home_win_probability + away_win_probability - 1) < 1e-6),
    CONSTRAINT predictions_pregame        CHECK (tipoff_utc IS NULL OR created_at < tipoff_utc),
    CONSTRAINT predictions_actual_is_team CHECK (actual_winner IS NULL OR actual_winner IN (home_team_id, away_team_id)),
    CONSTRAINT predictions_result_pair    CHECK ((actual_winner IS NULL) = (completed_at IS NULL)),
    -- A result cannot be recorded before a game could plausibly have ended.
    CONSTRAINT predictions_completed_after_tipoff CHECK (
        completed_at IS NULL OR tipoff_utc IS NULL OR completed_at >= tipoff_utc + interval '90 minutes')
);
CREATE INDEX predictions_game_date_idx ON public.predictions (game_date);
CREATE INDEX predictions_pending_idx   ON public.predictions (game_date) WHERE actual_winner IS NULL;
CREATE UNIQUE INDEX predictions_matchup_uidx ON public.predictions (game_date, home_team, away_team);

-- 2. Recovered predictions (exactly as displayed; unknowns stay NULL) -----------
CREATE TABLE public.predictions_recovered (
    recovery_id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    game_date              date             NOT NULL,
    away_team              text             NOT NULL,
    home_team              text             NOT NULL,
    predicted_winner       text             NOT NULL,
    home_win_probability   double precision NOT NULL,   -- as displayed (0.1% precision)
    away_win_probability   double precision NOT NULL,
    probability_is_rounded boolean          NOT NULL DEFAULT true,
    created_at             timestamptz      NOT NULL,   -- displayed time, minute precision
    created_at_displayed   text             NOT NULL,   -- original string, preserved
    tipoff_displayed_utc   timestamptz      NOT NULL,
    tipoff_displayed       text             NOT NULL,
    model_version          text             NOT NULL,
    recovered_from         text             NOT NULL,
    recovered_at           timestamptz      NOT NULL DEFAULT now(),
    -- filled only by verify_recovered_game()
    game_id                text,
    home_team_id           text,
    away_team_id           text,
    verified_tipoff_utc    timestamptz,
    verified_at            timestamptz,
    -- filled only by grade_recovered_prediction() from a confirmed FINAL result
    actual_winner          text,
    completed_at           timestamptz,
    final_home_score       integer,
    final_away_score       integer,
    result_status          text,
    result_source          text,
    CONSTRAINT recovered_winner_is_team CHECK (predicted_winner IN (home_team, away_team)),
    CONSTRAINT recovered_prob_range     CHECK (home_win_probability BETWEEN 0 AND 1
                                               AND away_win_probability BETWEEN 0 AND 1),
    CONSTRAINT recovered_prob_sum       CHECK (abs(home_win_probability + away_win_probability - 1) <= 0.0011),
    CONSTRAINT recovered_displayed_pregame CHECK (created_at < tipoff_displayed_utc),
    CONSTRAINT recovered_verified_all_or_none CHECK (
        (verified_at IS NULL AND game_id IS NULL AND home_team_id IS NULL
            AND away_team_id IS NULL AND verified_tipoff_utc IS NULL)
     OR (verified_at IS NOT NULL AND game_id ~ '^[0-9]{10}$' AND home_team_id IS NOT NULL
            AND away_team_id IS NOT NULL AND home_team_id <> away_team_id
            AND verified_tipoff_utc IS NOT NULL AND created_at < verified_tipoff_utc)),
    CONSTRAINT recovered_actual_needs_verified CHECK (actual_winner IS NULL OR verified_at IS NOT NULL),
    CONSTRAINT recovered_actual_is_team CHECK (actual_winner IS NULL OR actual_winner IN (home_team_id, away_team_id)),
    CONSTRAINT recovered_result_complete CHECK (
        (actual_winner IS NULL AND completed_at IS NULL AND final_home_score IS NULL
            AND final_away_score IS NULL AND result_status IS NULL AND result_source IS NULL)
     OR (actual_winner IS NOT NULL AND completed_at IS NOT NULL
            AND final_home_score >= 0 AND final_away_score >= 0
            AND final_home_score <> final_away_score
            AND result_status = 'Final' AND length(btrim(result_source)) > 0
            AND ((actual_winner = home_team_id) = (final_home_score > final_away_score))
            AND completed_at >= verified_tipoff_utc + interval '90 minutes'))
);
CREATE UNIQUE INDEX predictions_recovered_matchup_uidx ON public.predictions_recovered (game_date, home_team, away_team);
CREATE UNIQUE INDEX predictions_recovered_game_uidx    ON public.predictions_recovered (game_id) WHERE game_id IS NOT NULL;

-- 3. Integrity triggers ----------------------------------------------------------
CREATE FUNCTION public.predictions_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'predictions rows cannot be deleted'; END IF;
    IF (NEW.game_id, NEW.game_date, NEW.tipoff_utc, NEW.home_team_id, NEW.home_team,
        NEW.away_team_id, NEW.away_team, NEW.predicted_winner, NEW.home_win_probability,
        NEW.away_win_probability, NEW.win_probability, NEW.created_at,
        NEW.model_version, NEW.rating_source)
       IS DISTINCT FROM
       (OLD.game_id, OLD.game_date, OLD.tipoff_utc, OLD.home_team_id, OLD.home_team,
        OLD.away_team_id, OLD.away_team, OLD.predicted_winner, OLD.home_win_probability,
        OLD.away_win_probability, OLD.win_probability, OLD.created_at,
        OLD.model_version, OLD.rating_source) THEN
        RAISE EXCEPTION 'original prediction fields are immutable';
    END IF;
    IF OLD.actual_winner IS NOT NULL AND (NEW.actual_winner IS DISTINCT FROM OLD.actual_winner
        OR NEW.completed_at IS DISTINCT FROM OLD.completed_at) THEN
        RAISE EXCEPTION 'result already recorded';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER predictions_guard_trg BEFORE UPDATE OR DELETE ON public.predictions
    FOR EACH ROW EXECUTE FUNCTION public.predictions_guard();

CREATE FUNCTION public.recovered_guard() RETURNS trigger
LANGUAGE plpgsql SET search_path = pg_catalog, public AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN RAISE EXCEPTION 'recovered rows cannot be deleted'; END IF;
    IF (NEW.game_date, NEW.away_team, NEW.home_team, NEW.predicted_winner, NEW.home_win_probability,
        NEW.away_win_probability, NEW.probability_is_rounded, NEW.created_at, NEW.created_at_displayed,
        NEW.tipoff_displayed_utc, NEW.tipoff_displayed, NEW.model_version, NEW.recovered_from, NEW.recovered_at)
       IS DISTINCT FROM
       (OLD.game_date, OLD.away_team, OLD.home_team, OLD.predicted_winner, OLD.home_win_probability,
        OLD.away_win_probability, OLD.probability_is_rounded, OLD.created_at, OLD.created_at_displayed,
        OLD.tipoff_displayed_utc, OLD.tipoff_displayed, OLD.model_version, OLD.recovered_from, OLD.recovered_at) THEN
        RAISE EXCEPTION 'recovered original fields are immutable';
    END IF;
    IF OLD.verified_at IS NOT NULL AND (NEW.game_id, NEW.home_team_id, NEW.away_team_id,
        NEW.verified_tipoff_utc, NEW.verified_at)
       IS DISTINCT FROM (OLD.game_id, OLD.home_team_id, OLD.away_team_id,
        OLD.verified_tipoff_utc, OLD.verified_at) THEN
        RAISE EXCEPTION 'game identity already verified';
    END IF;
    IF OLD.actual_winner IS NOT NULL AND (NEW.actual_winner, NEW.completed_at, NEW.final_home_score,
        NEW.final_away_score, NEW.result_status, NEW.result_source)
       IS DISTINCT FROM (OLD.actual_winner, OLD.completed_at, OLD.final_home_score,
        OLD.final_away_score, OLD.result_status, OLD.result_source) THEN
        RAISE EXCEPTION 'result already recorded';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER recovered_guard_trg BEFORE UPDATE OR DELETE ON public.predictions_recovered
    FOR EACH ROW EXECUTE FUNCTION public.recovered_guard();

-- Cross-table duplicate prevention (same game_id OR same date + teams).
-- The advisory lock serialises concurrent inserts of the same matchup.
CREATE FUNCTION public.prediction_cross_duplicate_guard() RETURNS trigger
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE clash boolean;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(NEW.game_date::text || '|' || NEW.home_team || '|' || NEW.away_team, 0));
    IF TG_TABLE_NAME = 'predictions' THEN
        SELECT EXISTS (SELECT 1 FROM public.predictions_recovered r
                       WHERE r.game_id = NEW.game_id
                          OR (r.game_date = NEW.game_date AND r.home_team = NEW.home_team AND r.away_team = NEW.away_team))
          INTO clash;
    ELSE
        SELECT EXISTS (SELECT 1 FROM public.predictions p
                       WHERE (NEW.game_id IS NOT NULL AND p.game_id = NEW.game_id)
                          OR (p.game_date = NEW.game_date AND p.home_team = NEW.home_team AND p.away_team = NEW.away_team))
          INTO clash;
    END IF;
    IF clash THEN
        RAISE EXCEPTION 'duplicate game across predictions and predictions_recovered';
    END IF;
    RETURN NEW;
END $$;
CREATE TRIGGER predictions_xdup_trg BEFORE INSERT ON public.predictions
    FOR EACH ROW EXECUTE FUNCTION public.prediction_cross_duplicate_guard();
CREATE TRIGGER recovered_xdup_trg BEFORE INSERT OR UPDATE OF game_id ON public.predictions_recovered
    FOR EACH ROW EXECUTE FUNCTION public.prediction_cross_duplicate_guard();

-- 4. Secure verification / grading (the app role cannot UPDATE the recovered table) ---
CREATE FUNCTION public.verify_recovered_game(
    p_recovery_id bigint, p_game_id text, p_home_team_id text, p_away_team_id text, p_tipoff_utc timestamptz)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE r public.predictions_recovered%ROWTYPE;
BEGIN
    SELECT * INTO r FROM public.predictions_recovered WHERE recovery_id = p_recovery_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'recovered prediction % not found', p_recovery_id; END IF;
    IF r.verified_at IS NOT NULL THEN RAISE EXCEPTION 'already verified'; END IF;
    IF p_game_id !~ '^[0-9]{10}$' THEN RAISE EXCEPTION 'game_id must be 10 digits'; END IF;
    IF r.created_at >= p_tipoff_utc THEN RAISE EXCEPTION 'prediction was not made before official tipoff'; END IF;
    UPDATE public.predictions_recovered
       SET game_id = p_game_id, home_team_id = p_home_team_id, away_team_id = p_away_team_id,
           verified_tipoff_utc = p_tipoff_utc, verified_at = now()
     WHERE recovery_id = p_recovery_id;
END $$;

-- Grades only from a confirmed FINAL result: NBA gameStatus 3, final scores that
-- agree with the winner, a named source, and >= 90 minutes after official tipoff.
CREATE FUNCTION public.grade_recovered_prediction(
    p_recovery_id bigint, p_actual_winner_id text,
    p_home_score integer, p_away_score integer, p_game_status integer, p_result_source text)
RETURNS void LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public AS $$
DECLARE r public.predictions_recovered%ROWTYPE;
BEGIN
    SELECT * INTO r FROM public.predictions_recovered WHERE recovery_id = p_recovery_id FOR UPDATE;
    IF NOT FOUND THEN RAISE EXCEPTION 'recovered prediction % not found', p_recovery_id; END IF;
    IF r.verified_at IS NULL THEN RAISE EXCEPTION 'verify the game identity first'; END IF;
    IF r.actual_winner IS NOT NULL THEN RAISE EXCEPTION 'result already recorded'; END IF;
    IF p_game_status IS DISTINCT FROM 3 THEN RAISE EXCEPTION 'game is not final (gameStatus must be 3)'; END IF;
    IF p_home_score IS NULL OR p_away_score IS NULL OR p_home_score < 0 OR p_away_score < 0
       OR p_home_score = p_away_score THEN
        RAISE EXCEPTION 'final scores are required and cannot be tied'; END IF;
    IF p_actual_winner_id NOT IN (r.home_team_id, r.away_team_id) THEN
        RAISE EXCEPTION 'winner is not one of the two teams'; END IF;
    IF (p_actual_winner_id = r.home_team_id) <> (p_home_score > p_away_score) THEN
        RAISE EXCEPTION 'winner does not match the final score'; END IF;
    IF p_result_source IS NULL OR length(btrim(p_result_source)) = 0 THEN
        RAISE EXCEPTION 'result source is required'; END IF;
    IF now() < r.verified_tipoff_utc + interval '90 minutes' THEN
        RAISE EXCEPTION 'too soon after tipoff for a final result'; END IF;
    UPDATE public.predictions_recovered
       SET actual_winner = p_actual_winner_id, completed_at = now(),
           final_home_score = p_home_score, final_away_score = p_away_score,
           result_status = 'Final', result_source = btrim(p_result_source)
     WHERE recovery_id = p_recovery_id;
END $$;

-- 5. Unified history and cumulative accuracy ---------------------------------------
CREATE VIEW public.prediction_history_all WITH (security_invoker = true) AS
SELECT 'pregame_saved'::text AS source, game_id, game_date, tipoff_utc,
       home_team_id, home_team, away_team_id, away_team, predicted_winner,
       home_win_probability, away_win_probability, win_probability,
       false AS probability_is_rounded, created_at, model_version, rating_source,
       actual_winner, completed_at, true AS identity_verified,
       CASE WHEN actual_winner IS NULL THEN NULL
            ELSE predicted_winner = CASE WHEN actual_winner = home_team_id THEN home_team ELSE away_team END
       END AS correct
FROM public.predictions
UNION ALL
SELECT 'recovered', game_id, game_date, COALESCE(verified_tipoff_utc, tipoff_displayed_utc),
       home_team_id, home_team, away_team_id, away_team, predicted_winner,
       home_win_probability, away_win_probability,
       GREATEST(home_win_probability, away_win_probability),
       probability_is_rounded, created_at, model_version, NULL::text,
       actual_winner, completed_at, verified_at IS NOT NULL,
       CASE WHEN actual_winner IS NULL THEN NULL
            ELSE predicted_winner = CASE WHEN actual_winner = home_team_id THEN home_team ELSE away_team END
       END
FROM public.predictions_recovered;

-- Recovered rows have actual_winner only after verification + confirmed final
-- result (table constraints), so they cannot count toward accuracy before that.
CREATE VIEW public.prediction_accuracy WITH (security_invoker = true) AS
SELECT count(*)                                          AS total_predictions,
       count(*) FILTER (WHERE actual_winner IS NOT NULL) AS completed,
       count(*) FILTER (WHERE actual_winner IS NULL)     AS pending,
       count(*) FILTER (WHERE correct)                   AS correct,
       count(*) FILTER (WHERE correct IS FALSE)          AS incorrect,
       round((count(*) FILTER (WHERE correct))::numeric
             / NULLIF(count(*) FILTER (WHERE actual_winner IS NOT NULL), 0), 4) AS accuracy
FROM public.prediction_history_all;

-- 6. Application role (NO password in this script) ----------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'courtvision_app') THEN
        CREATE ROLE courtvision_app LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION
            NOBYPASSRLS NOINHERIT CONNECTION LIMIT 5;
    END IF;
END $$;
-- Set the password out-of-band:  \password courtvision_app   (or Supabase dashboard).
ALTER ROLE courtvision_app SET statement_timeout = '15s';
ALTER ROLE courtvision_app SET search_path = public;

-- 7. Privileges ---------------------------------------------------------------------
REVOKE ALL ON public.predictions, public.predictions_recovered,
              public.prediction_history_all, public.prediction_accuracy
    FROM PUBLIC, anon, authenticated, courtvision_app;
REVOKE ALL ON FUNCTION public.predictions_guard(), public.recovered_guard(),
    public.prediction_cross_duplicate_guard(),
    public.verify_recovered_game(bigint, text, text, text, timestamptz),
    public.grade_recovered_prediction(bigint, text, integer, integer, integer, text)
    FROM PUBLIC, anon, authenticated;
REVOKE CREATE ON SCHEMA public FROM PUBLIC, anon, authenticated, courtvision_app;
GRANT USAGE ON SCHEMA public TO courtvision_app;

GRANT SELECT, INSERT ON public.predictions TO courtvision_app;
GRANT UPDATE (actual_winner, completed_at) ON public.predictions TO courtvision_app;
GRANT SELECT ON public.predictions_recovered, public.prediction_history_all,
                public.prediction_accuracy TO courtvision_app;
GRANT EXECUTE ON FUNCTION public.verify_recovered_game(bigint, text, text, text, timestamptz),
    public.grade_recovered_prediction(bigint, text, integer, integer, integer, text) TO courtvision_app;
-- Not granted: DELETE, TRUNCATE, DDL, any direct write on predictions_recovered.

-- 8. Row Level Security -------------------------------------------------------------
ALTER TABLE public.predictions           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions           FORCE  ROW LEVEL SECURITY;
ALTER TABLE public.predictions_recovered ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.predictions_recovered FORCE  ROW LEVEL SECURITY;

CREATE POLICY owner_all ON public.predictions           TO postgres USING (true) WITH CHECK (true);
CREATE POLICY owner_all ON public.predictions_recovered TO postgres USING (true) WITH CHECK (true);

CREATE POLICY app_select     ON public.predictions FOR SELECT TO courtvision_app USING (true);
CREATE POLICY app_insert     ON public.predictions FOR INSERT TO courtvision_app
    WITH CHECK (actual_winner IS NULL AND completed_at IS NULL);
CREATE POLICY app_set_result ON public.predictions FOR UPDATE TO courtvision_app
    USING (actual_winner IS NULL) WITH CHECK (actual_winner IS NOT NULL AND completed_at IS NOT NULL);
CREATE POLICY app_select     ON public.predictions_recovered FOR SELECT TO courtvision_app USING (true);

COMMIT;
