import kfp
from kfp import dsl, compiler
from kfp.client import Client
import yaml

@dsl.component(base_image='python:3.10-slim')
def scrape_component():
    import subprocess
    subprocess.run(['python', '-m', 'agents.scraper_orchestrator'])

@dsl.component(base_image='python:3.10-slim')
def ml_analysis_component():
    import subprocess
    subprocess.run(['python', '-m', 'ml.pipeline'])

@dsl.component(base_image='python:3.10-slim')
def enrichment_component():
    import subprocess
    subprocess.run(['python', '-m', 'llm.enrich'])

@dsl.pipeline(name='Smart eCommerce Pipeline', description='Pipeline complet avec scraping, ML et LLM')
def smart_ecommerce_pipeline():
    scrape_task = scrape_component()
    ml_task = ml_analysis_component().after(scrape_task)
    enrich_task = enrichment_component().after(ml_task)

if __name__ == '__main__':
    # Compiler le pipeline
    compiler.Compiler().compile(smart_ecommerce_pipeline, 'smart_ecommerce_pipeline.yaml')
    
    # Connexion au cluster Kubeflow (Minikube) et exécution
    client = Client(host='http://localhost:8080')
    experiment = client.create_experiment(name='SmartEcommerce')
    run = client.run_pipeline(experiment.id, 'smart-ecommerce-run', 'smart_ecommerce_pipeline.yaml')
    print(f"Pipeline exécuté : {run.id}")