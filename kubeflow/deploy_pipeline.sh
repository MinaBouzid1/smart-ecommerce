#!/bin/bash
# Déploiement d'un pipeline Kubeflow sur Minikube

echo "Démarrage de Minikube avec KubeFlow..."
minikube start --cpus=4 --memory=8192 --disk-size=20g
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.5"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic-pns?ref=2.0.5"

# Attente que le service ml-pipeline-ui soit prêt
kubectl wait --for=condition=ready pod -l app=ml-pipeline-ui --timeout=300s -n kubeflow

# Port-forward pour accéder à l'UI
echo "Lancement du port-forward vers le dashboard Kubeflow (accès via http://localhost:8080)"
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80