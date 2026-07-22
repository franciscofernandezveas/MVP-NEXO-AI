-- Datos históricos limpios y normalizados
create table sales_clean (
    id bigint generated always as identity primary key,
    fecha date not null,
    sede text not null,
    producto text not null,
    cantidad numeric(12,2) not null default 0,
    precio_promedio numeric(12,2),
    created_at timestamptz default now(),
    unique(fecha, sede, producto)
);

create index idx_sales_clean_lookup on sales_clean(fecha, sede, producto);
create index idx_sales_clean_sede on sales_clean(sede, producto, fecha);

-- Predicciones generadas
create table demand_forecasts (
    id bigint generated always as identity primary key,
    fecha date not null,
    sede text not null,
    producto text not null,
    prediccion integer not null,
    prediccion_con_buffer integer,
    tipo text not null check (tipo in ('historica','futura')),
    modelo_version text not null,
    fecha_generacion timestamptz not null default now(),
    metricas jsonb,
    dias_pronosticados integer default 1,
    safety_stock numeric(10,2),
    raw_context jsonb
);

create index idx_demand_forecasts_lookup on demand_forecasts(sede, producto, fecha desc);

-- Artefactos de modelos entrenados
create table model_artifacts (
    id bigint generated always as identity primary key,
    producto text not null,
    sede text not null,
    modelo_version text not null unique,
    entrenado_hasta date not null,
    features jsonb not null,
    le_prod jsonb not null,
    le_sede jsonb not null,
    safety_stock numeric(10,2),
    metricas jsonb,
    test_days int,
    ruta_artifact text,
    created_at timestamptz default now()
);

create unique index idx_model_artifacts_prod_sede
    on model_artifacts(producto, sede, modelo_version desc);
