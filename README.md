# Chest-Cancer-Classfication


## Workflow
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src/config
6. Update the components
7. Update the pipeline
8. Update main.py
9. Update dvc.yaml

## Run
1. Train, evaluate, and log metrics with `python main.py`
2. Start the website with `python app.py`
3. Open the local Flask URL shown in the terminal to upload a CT image and view the prediction