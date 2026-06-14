# irrepx Technical Design Documents

**Quick Links**:
- 📋 [TODO.md](../TODO.md) — v0.1.0 task list
- 🏗 [architecture.md](architecture.md) — System architecture
- 🛠 [development.md](development.md) — Development guide
- ❓ [FAQ.md](FAQ.md) — Common issues & gotchas
- 🔒 `../DEVELOPMENT.md` — Internal roadmap (gitignored, DO NOT COMMIT)

## Document Organization

```
design/
├── README.md           ← you are here
├── architecture.md     — Package structure, data flow, design decisions
├── development.md      — Build system, testing, conventions
└── FAQ.md              — CG normalization, gate logic, env tips
```

## Quick Navigation

| Category | Document | Section |
|----------|----------|---------|
| Package layout | architecture.md | [Package Structure](#) |
| Dual-mode design | architecture.md | [Dual-Mode Strategy](#) |
| CG coefficients | FAQ.md | [CG normalization](#) |
| JAX versions | FAQ.md | [JAX version pinning](#) |
| Build & test | development.md | [Make Targets](#) |
| Code style | development.md | [Conventions](#) |

## Status Legend

✅ Implemented | 🚧 In Progress | 🔜 Planned
