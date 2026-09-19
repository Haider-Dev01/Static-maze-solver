# Contribuer

## Installation de développement

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Vérifications avant une pull request

```bash
ruff format .
ruff check .
mypy
pytest
```

Une contribution doit :

- inclure des tests pour tout nouveau comportement ;
- conserver une couverture globale d'au moins 80 % ;
- respecter l'API Gymnasium ;
- rester reproductible avec une seed explicite ;
- mettre à jour le README ou le changelog si l'interface publique change.

Les commits doivent être courts, descriptifs et limités à une évolution cohérente.
