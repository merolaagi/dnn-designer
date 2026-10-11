-- A read-only login for the Agent lab's Station data tools, on the c-store platform's database.
--
--   psql -p 5432 -d cstore_platform -v pw="$PW" -v tid='TENANT_ID' -f station_reader.sql
--
-- It can read the operational tables (sales, fuel, tanks, deliveries, prices, items, purchasing,
-- shifts, days, lottery, ingestion) and nothing else: no users, tokens, bank, billing or POS
-- connector settings; no site API keys and no raw XML. Every session it opens is read-only, gives up
-- on a query after 10 seconds, and sees only the one tenant named here (the tables' row-level
-- security reads app.current_tenant_id). Running this again resets the password and grants.

DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'agent_reader') THEN
    CREATE ROLE agent_reader LOGIN;
  END IF;
END $$;

ALTER ROLE agent_reader WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS PASSWORD :'pw';
ALTER ROLE agent_reader SET default_transaction_read_only = on;
ALTER ROLE agent_reader SET statement_timeout = '10s';
ALTER ROLE agent_reader SET app.current_tenant_id = :'tid';

REVOKE ALL ON ALL TABLES IN SCHEMA public FROM agent_reader;
GRANT CONNECT ON DATABASE cstore_platform TO agent_reader;
GRANT USAGE ON SCHEMA public TO agent_reader;

-- the site list without its API key, and file ingestion without the raw XML
GRANT SELECT (id, tenant_id, name, address, is_active, created_at, auto_close_hours_after_last_shift) ON sites TO agent_reader;
GRANT SELECT (id, site_id, filename, pos_brand, file_type, processed, received_at) ON xml_files TO agent_reader;

GRANT SELECT ON
  agent_pings, day_reports, day_cash_summary, day_close_entries, day_mop_totals, day_reconciliation,
  shift_reports, shift_reconciliations, shift_discounts, cashier_stats, department_totals, dept_margins,
  tender_totals, tender_breakdowns, network_totals, tax_collections,
  fuel_grades, fuel_grade_sales, fuel_sale_totals, fuel_inside_outside_per_grade, fuel_inventory_daily,
  fuel_tanks, grade_tank_blends, fuel_deliveries, fuel_price_changes, fuel_costs, fuel_reconciliations,
  naxml_mcm_daily, naxml_ism_daily, merchandise_code_attrs, items, item_categories, inventory_levels,
  inventory_adjustments, price_changes, promotions, promotion_items, purchase_orders, purchase_order_lines,
  receiving_docs, receiving_doc_lines, vendors, invoices, expenses,
  lottery_games, lottery_packs, lottery_counts, lottery_payouts_daily, digest_summaries
TO agent_reader;

\echo 'agent_reader is ready: read-only, 10-second queries, one tenant.'
