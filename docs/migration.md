# FreeToken to SparkLab migration

SparkLab is being introduced as a product layer without forcing an immediate
engine-package rename.

| Existing surface | SparkLab surface | Current behavior |
|---|---|---|
| `ft` | `sparklab` | Both commands are installed and supported |
| `freetoken.*` imports | unchanged | The engine package remains `freetoken` |
| `FREETOKEN_*` | `SPARKLAB_*` | Product name wins where migrated; legacy value is the fallback |
| `~/.cache/freetoken` | `~/.cache/sparklab` | New bandwidth profiles use SparkLab; old profiles are discovered read-only |
| API routes and schemas | unchanged | Existing OpenAI and Anthropic clients remain compatible |
| FreeToken paper name | unchanged | Academic attribution remains FreeToken |

The first SparkLab product commands are:

```bash
sparklab doctor
sparklab models
sparklab status
```

Engine commands can be switched mechanically:

```bash
ft serve --model MODEL        # remains valid
sparklab serve --model MODEL  # same engine path
```

Do not rename Python imports in downstream integrations yet. A package-level
migration, if justified after product validation, will be a separate major-version
decision with an explicit deprecation period.
