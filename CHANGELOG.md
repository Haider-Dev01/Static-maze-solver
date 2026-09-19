# Changelog

Toutes les évolutions notables du projet sont documentées ici.

## [1.0.0] - 2026-09-19

### Ajouté

- environnement Gymnasium validé avec limite de pas ;
- agent Q-learning reproductible et persistance JSON ;
- commandes `train`, `evaluate`, `demo` et `benchmark` ;
- presets et génération de labyrinthes solvables ;
- baseline optimale A* ;
- métriques CSV/JSON et courbes de convergence ;
- rendu Pygame enrichi et exports GIF/PNG ;
- suite de tests, couverture, Ruff, mypy et GitHub Actions ;
- documentation portfolio et benchmark multi-seed.

### Modifié

- remplacement de l'ancienne API Gym par Gymnasium ;
- séparation stricte de l'entraînement et de l'évaluation gloutonne ;
- réorganisation du code en package `src/static_maze_solver`.

### Supprimé

- obstacle invalide situé hors de la grille ;
- dépendance à un pickle historique incompatible ;
- épisodes sans limite de pas.
