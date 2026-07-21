create table if not exists demo_sessions (
  id uuid primary key default gen_random_uuid(),
  slug text unique,
  profile jsonb default '{}'::jsonb,
  is_demo boolean default false,
  created_at timestamptz default now()
);

create table if not exists session_credentials (
  session_id uuid references demo_sessions(id) primary key,
  is_demo boolean default false,
  openai_key_encrypted text,
  db_config_encrypted text,
  updated_at timestamptz default now()
);

create table if not exists agent_docs (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references demo_sessions(id),
  content text,
  version int default 1,
  created_at timestamptz default now()
);

create table if not exists feedback (
  id uuid primary key default gen_random_uuid(),
  session_id uuid references demo_sessions(id),
  message_id uuid,
  rating int,
  comment text,
  created_at timestamptz default now()
);
