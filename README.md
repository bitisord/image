# Traitement image

Cette implémentation fournit un pipeline complet pour extraire les informations
présentes sur une étiquette produit à partir d’une photo grâce à un modèle LLM
multimodal, puis pour stocker les données extraites dans une base H2.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Téléchargez le driver JDBC H2 (fichier `.jar`) depuis
[https://www.h2database.com](https://www.h2database.com) et placez-le dans un
répertoire local, par exemple `./drivers/h2-2.2.224.jar`.

## Utilisation

1. Exportez votre clé API :
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
2. Lancez l’ingestion :
   ```bash
   python -m multimodal_pipeline \
     --image-path ./data/photo_produit.jpg \
     --jdbc-url "jdbc:h2:~/products" \
     --jdbc-driver ./drivers/h2-2.2.224.jar \
     --model gpt-4o-mini
   ```

Le script va :

* Encoder la photo en base64 et l’envoyer au modèle multimodal (GPT-4o mini par
  défaut) en lui demandant un JSON structuré.
* Normaliser la réponse et créer la table `product_labels` dans H2 si nécessaire.
* Insérer ou mettre à jour l’enregistrement correspondant au SKU dans la base.

Le programme affiche également le JSON final sur la sortie standard afin de
faciliter le débogage ou l’intégration dans un pipeline existant.

## Personnalisation

* **Prompt** : vous pouvez fournir un fichier prompt personnalisé via
  `--prompt ./prompts/label.txt`.
* **Journalisation** : contrôlez le niveau avec `--log-level DEBUG` pour suivre
  les étapes de la requête et du stockage.

## Notes

* Le module `jaydebeapi` nécessite Java installé pour charger le driver JDBC.
* Vous pouvez appeler cette logique depuis un orchestrateur (Airflow, Prefect,
  etc.) en important `MultimodalLabelIngestor` depuis `multimodal_pipeline`.
